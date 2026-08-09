"""
Categorical Midpoint Generative Model (MGM) for fixed-length text sequences.

This is the categorical counterpart of the continuous MGM training script:

  real endpoint       R : one-hot (optionally noised) token distributions
  generated endpoint  P : softmax-relaxed token distributions
  displacement        D : R - P
  hidden observation  Y : tokenwise samples from R or P, using one of the
                          two symmetric mixing probabilities t and 1-t

The hidden branch and token masks are never shown to the critic.

Two critic objectives are available:

  --critic-loss mse
      Regress f(Y,t) directly toward R-P.

  --critic-loss ce
      Train two denoisers with soft-label cross entropy:
        q_real(Y,t) -> R,  q_fake(Y,t) -> P,
      then define f(Y,t) = q_real(Y,t) - q_fake(Y,t).
      At the population optimum this is E[R-P | Y,t], the same field learned
      by the MSE critic.

The generator always minimizes the original MGM payoff

    E[ 2 <f(Y,t), R-P> - ||f(Y,t)||^2 ].

Two relaxation gradients are available:

  --gen-grad stopped   Stable biased semi-gradient.  Stop P -> Y -> critic,
                       but retain the direct gradient through R-P.
  --gen-grad full      Differentiate through both R-P and P -> Y -> critic.

The default corpus is deliberately tiny and built in so the script runs
without downloads.  For an actual dataset, pass a UTF-8 text file containing
one training sequence per line:

    python categorical_mgm.py --data-path corpus.txt --critic-loss ce

This example uses characters as categories.  Replacing CharCorpus with a
subword tokenizer does not change the MGM objectives.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TEXTS = (
    "a small bird sings.",
    "a small bird waits.",
    "a small bird flies.",
    "the moon is bright.",
    "the moon is silver.",
    "the moon is quiet.",
    "the stars are bright.",
    "the stars are quiet.",
    "the stars are distant.",
    "the river runs slowly.",
    "the river runs softly.",
    "the river reflects light.",
    "a warm wind rises.",
    "a warm wind passes.",
    "a cool wind returns.",
    "green leaves move.",
    "green leaves shimmer.",
    "gold leaves fall.",
    "morning light arrives.",
    "morning rain arrives.",
    "evening light fades.",
    "soft rain begins.",
    "soft rain continues.",
    "soft rain ends.",
    "the garden is still.",
    "the garden is green.",
    "the garden smells sweet.",
    "clouds cross the sky.",
    "birds cross the sky.",
    "light fills the sky.",
)


@dataclass
class Config:
    data_path: str | None = None
    run_dir: str = "experiments/categorical_mgm"
    max_seq_len: int = 32

    noise_dim: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.0

    batch_size: int = 128
    n_steps: int = 100_000
    critic_steps_per_gen: int = 1
    lr_critic: float = 1e-4
    lr_gen: float = 1e-4
    grad_clip: float = 1.0

    critic_loss: str = "ce"       # "ce" or "mse"
    gen_grad: str = "full"     # "stopped" or "full"

    # Real endpoint noise.  "smooth" keeps the endpoint on the simplex;
    # "replace" randomly replaces tokens and keeps the endpoint one-hot.
    real_noise: str = "smooth"    # "none", "smooth", or "replace"
    real_noise_eps: float = 0.01

    # P = softmax(logits / temperature).  Equal endpoints give a fixed temp.
    temperature_start: float = 2.0
    temperature_end: float = 2.0

    # Sample t in [t_min, t_max].  The second hidden branch uses 1-t.
    t_min: float = 0.0
    t_max: float = 0.5
    rho: float = 1.0

    seed: int = 42
    device: str = "auto"
    log_every: int = 100
    sample_every: int = 1_000
    n_samples: int = 12


class CharCorpus:
    """Fixed-length character sequences represented internally by token IDs."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, texts: Sequence[str], max_seq_len: int):
        if max_seq_len < 3:
            raise ValueError("max_seq_len must be at least 3")
        texts = [text.rstrip("\n") for text in texts if text.rstrip("\n")]
        if not texts:
            raise ValueError("The corpus contains no non-empty lines")

        characters = sorted(set("".join(texts)))
        self.itos = [self.PAD, self.BOS, self.EOS, *characters]
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.max_seq_len = max_seq_len
        self.tokens = torch.tensor(
            [self.encode(text) for text in texts], dtype=torch.long
        )

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        content = [self.stoi[ch] for ch in text[: self.max_seq_len - 2]]
        ids = [self.stoi[self.BOS], *content, self.stoi[self.EOS]]
        ids += [self.stoi[self.PAD]] * (self.max_seq_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        result: list[str] = []
        for index in ids:
            token = self.itos[int(index)]
            if token == self.EOS:
                break
            if token not in (self.PAD, self.BOS):
                result.append(token)
        return "".join(result)


def transformer_stack(
    d_model: int, n_heads: int, n_layers: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class NonlinearHead(nn.Module):
    """A small nonlinear vocabulary head; the transformer remains shared."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class CategoricalGenerator(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()

        self.seq_len = cfg.max_seq_len
        self.d_model = cfg.d_model

        self.z_projection = nn.Sequential(
            nn.Linear(
                cfg.noise_dim,
                2 * cfg.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                2 * cfg.d_model,
                cfg.max_seq_len * cfg.d_model,
            ),
        )

        self.position = nn.Parameter(
            torch.randn(
                1,
                cfg.max_seq_len,
                cfg.d_model,
            ) / math.sqrt(cfg.d_model)
        )

        self.backbone = transformer_stack(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.dropout,
        )

        self.output_head = NonlinearHead(
            cfg.d_model,
            vocab_size,
        )

    def logits(self, z):
        batch_size = z.shape[0]

        latent_sequence = self.z_projection(z).reshape(
            batch_size,
            self.seq_len,
            self.d_model,
        )

        hidden = latent_sequence + self.position
        hidden = self.backbone(hidden)
        return self.output_head(hidden)

    def relaxed(self, z: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.softmax(self.logits(z) / temperature, dim=-1)


class CategoricalCritic(nn.Module):
    """Shared nonlinear encoder with an MSE field head or two CE denoisers."""

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.kind = cfg.critic_loss
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position = nn.Parameter(
            torch.randn(1, cfg.max_seq_len, cfg.d_model) / math.sqrt(cfg.d_model)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.backbone = transformer_stack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout
        )

        if self.kind == "mse":
            self.field_head = NonlinearHead(cfg.d_model, vocab_size)
        elif self.kind == "ce":
            self.real_head = NonlinearHead(cfg.d_model, vocab_size)
            self.fake_head = NonlinearHead(cfg.d_model, vocab_size)
        else:
            raise ValueError(f"Unknown critic loss: {self.kind}")

    def encode(self, observation: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Expected embedding allows soft observations and preserves gradients.
        hidden = observation @ self.token_embedding.weight
        time = self.time_embedding(t.reshape(t.shape[0], 1)).unsqueeze(1)
        hidden = hidden + self.position + time
        return self.backbone(hidden)

    def forward(self, observation: torch.Tensor, t: torch.Tensor):
        hidden = self.encode(observation, t)
        if self.kind == "mse":
            return self.field_head(hidden)
        return self.real_head(hidden), self.fake_head(hidden)

    def field(self, observation: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the conditional displacement estimate f(Y,t)."""
        output = self(observation, t)
        if self.kind == "mse":
            return output
        real_logits, fake_logits = output
        q_real = F.softmax(real_logits, dim=-1)
        q_fake = F.softmax(fake_logits, dim=-1)
        return q_real - q_fake


def load_texts(data_path: str | None) -> list[str]:
    if data_path is None:
        return list(DEFAULT_TEXTS)
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(cfg: Config) -> None:
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0.0 <= cfg.t_min <= cfg.t_max <= 0.5:
        raise ValueError("Require 0 <= t_min <= t_max <= 0.5")
    if cfg.rho <= 0.0:
        raise ValueError("rho must be positive")
    if not 0.0 <= cfg.real_noise_eps <= 1.0:
        raise ValueError("real_noise_eps must be in [0,1]")
    if cfg.temperature_start <= 0.0 or cfg.temperature_end <= 0.0:
        raise ValueError("Relaxation temperatures must be positive")
    if cfg.n_steps < 1 or cfg.critic_steps_per_gen < 1:
        raise ValueError("Training step counts must be positive")


def sample_real_endpoint(
    token_pool: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    noise_mode: str,
    noise_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token IDs and construct a one-hot/noised simplex endpoint R."""
    indices = torch.randint(0, token_pool.shape[0], (batch_size,), device=token_pool.device)
    token_ids = token_pool[indices]

    if noise_mode == "replace" and noise_eps > 0.0:
        replace = torch.rand_like(token_ids, dtype=torch.float32) < noise_eps
        random_ids = torch.randint_like(token_ids, high=vocab_size)
        token_ids = torch.where(replace, random_ids, token_ids)

    endpoint = F.one_hot(token_ids, num_classes=vocab_size).float()
    if noise_mode == "smooth" and noise_eps > 0.0:
        endpoint = (1.0 - noise_eps) * endpoint + noise_eps / vocab_size
    elif noise_mode not in ("none", "replace", "smooth"):
        raise ValueError(f"Unknown real noise mode: {noise_mode}")

    return token_ids, endpoint


def sample_time(cfg: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    """Power-shaped sampling on [t_min,t_max], matching the continuous script."""
    low = cfg.t_min ** (1.0 / cfg.rho)
    high = cfg.t_max ** (1.0 / cfg.rho)
    uniform = torch.rand(batch_size, 1, 1, device=device)
    return (uniform * (high - low) + low) ** cfg.rho


def sample_hidden_observation(real, fake, cfg):
    batch_size = real.shape[0]
    t = sample_time(cfg, batch_size, real.device)

    branch = torch.rand(
        batch_size, 1, 1, device=real.device
    ) < 0.5

    real_weight = torch.where(
        branch,
        t,
        1.0 - t,
    )

    observation = (
        real_weight * real
        + (1.0 - real_weight) * fake
    )

    displacement = real - fake
    return observation, displacement, t

def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy supporting one-hot or soft simplex targets."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def critic_objective(
    critic: CategoricalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    observation, displacement, t = sample_hidden_observation(real, fake, cfg)
    output = critic(observation, t)

    if cfg.critic_loss == "mse":
        # Unconstrained regression field: f*(Y,t) = E[R-P | Y,t].
        return (output - displacement).square().sum(dim=-1).mean()

    real_logits, fake_logits = output
    # At optimum q_real=E[R|Y,t], q_fake=E[P|Y,t], hence q_real-q_fake=f*.
    return soft_cross_entropy(real_logits, real) + soft_cross_entropy(fake_logits, fake)


def generator_objective(
    generator: CategoricalGenerator,
    critic: CategoricalCritic,
    real: torch.Tensor,
    z: torch.Tensor,
    temperature: float,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    fake = generator.relaxed(z, temperature)

    # stopped: biased IDLM-style semi-gradient; full: pathwise soft relaxation.
    fake_for_observation = fake.detach() if cfg.gen_grad == "stopped" else fake
    observation, displacement, t = sample_hidden_observation(
        real, fake_for_observation, cfg
    )

    if cfg.gen_grad == "stopped":
        with torch.no_grad():
            field = critic.field(observation, t)
        # The direct displacement must still use the non-detached P.
        displacement = real - fake
    elif cfg.gen_grad == "full":
        field = critic.field(observation, t)
        displacement = real - fake
    else:
        raise ValueError(f"Unknown generator gradient mode: {cfg.gen_grad}")

    payoff = 2.0 * field * displacement - field.square()
    loss = payoff.sum(dim=-1).mean()
    return loss, fake


def temperature_at_step(cfg: Config, step: int) -> float:
    if cfg.n_steps == 1:
        return cfg.temperature_end
    fraction = (step - 1) / (cfg.n_steps - 1)
    # Geometric interpolation is natural for a positive scale.
    return cfg.temperature_start * (
        cfg.temperature_end / cfg.temperature_start
    ) ** fraction


@torch.no_grad()
def generate_text(
    generator: CategoricalGenerator,
    corpus: CharCorpus,
    cfg: Config,
    device: torch.device,
    n_samples: int,
) -> list[str]:
    was_training = generator.training
    generator.eval()
    z = torch.randn(n_samples, cfg.noise_dim, device=device)
    token_ids = generator.logits(z).argmax(dim=-1).cpu()
    texts = [corpus.decode(row.tolist()) for row in token_ids]
    generator.train(was_training)
    return texts


def write_samples(path: Path, step: int, samples: Sequence[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nstep {step}\n")
        for sample in samples:
            handle.write(sample + "\n")


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def train(cfg: Config) -> None:
    validate_config(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    texts = load_texts(cfg.data_path)
    corpus = CharCorpus(texts, cfg.max_seq_len)
    token_pool = corpus.tokens.to(device)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.txt"
    samples_path.write_text("", encoding="utf-8")

    saved_config = asdict(cfg)
    saved_config.update(
        device=str(device),
        n_real=len(corpus.tokens),
        vocab_size=corpus.vocab_size,
        vocabulary=corpus.itos,
    )
    (run_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generator = CategoricalGenerator(cfg, corpus.vocab_size).to(device)
    critic = CategoricalCritic(cfg, corpus.vocab_size).to(device)
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_gen, betas=(0.0, 0.99))
    opt_critic = torch.optim.AdamW(
        critic.parameters(), lr=cfg.lr_critic, betas=(0.0, 0.99)
    )

    history_critic: list[float] = []
    history_gen: list[float] = []
    start_time = time.time()

    print(
        f"device={device}  real={len(corpus.tokens)}  "
        f"shape=({cfg.max_seq_len},{corpus.vocab_size})  "
        f"critic={cfg.critic_loss}  gen_grad={cfg.gen_grad}"
    )
    print(f"{'step':>7}  {'L_critic':>10}  {'L_gen':>10}  {'H(P)':>8}  {'temp':>7}")

    for step in range(1, cfg.n_steps + 1):
        temperature = temperature_at_step(cfg, step)

        # Critic updates: endpoints are constants with respect to the critic.
        generator.eval()
        critic.train()
        for _ in range(cfg.critic_steps_per_gen):
            _, real = sample_real_endpoint(
                token_pool,
                cfg.batch_size,
                corpus.vocab_size,
                cfg.real_noise,
                cfg.real_noise_eps,
            )
            with torch.no_grad():
                z = torch.randn(cfg.batch_size, cfg.noise_dim, device=device)
                fake = generator.relaxed(z, temperature)

            loss_critic = critic_objective(critic, real, fake, cfg)
            opt_critic.zero_grad(set_to_none=True)
            loss_critic.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
            opt_critic.step()

        # Generator update: freeze critic weights, but in full mode retain the
        # differentiable path through the critic with respect to its input Y.
        generator.train()
        critic.eval()
        set_requires_grad(critic, False)
        _, real = sample_real_endpoint(
            token_pool,
            cfg.batch_size,
            corpus.vocab_size,
            cfg.real_noise,
            cfg.real_noise_eps,
        )
        z = torch.randn(cfg.batch_size, cfg.noise_dim, device=device)
        loss_gen, fake = generator_objective(
            generator, critic, real, z, temperature, cfg
        )

        opt_gen.zero_grad(set_to_none=True)
        loss_gen.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
        opt_gen.step()
        set_requires_grad(critic, True)

        entropy = -(fake.detach() * fake.detach().clamp_min(1e-8).log()).sum(-1).mean()
        history_critic.append(float(loss_critic.detach()))
        history_gen.append(float(loss_gen.detach()))

        if step == 1 or step % cfg.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"{step:7d}  {loss_critic.item():10.4f}  {loss_gen.item():10.4f}  "
                f"{entropy.item():8.4f}  {temperature:7.4f}  ({elapsed:.1f}s)"
            )

        if step == 1 or step % cfg.sample_every == 0:
            samples = generate_text(
                generator, corpus, cfg, device, cfg.n_samples
            )
            write_samples(samples_path, step, samples)
            print("samples:", " | ".join(repr(sample) for sample in samples[:4]))

    final_samples = generate_text(generator, corpus, cfg, device, cfg.n_samples)
    write_samples(samples_path, cfg.n_steps, final_samples)

    np.savez(
        run_dir / "losses.npz",
        critic=np.asarray(history_critic, dtype=np.float32),
        generator=np.asarray(history_gen, dtype=np.float32),
    )
    torch.save(
        {
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "config": saved_config,
            "vocabulary": corpus.itos,
        },
        run_dir / "checkpoint.pt",
    )
    print(f"Done in {time.time() - start_time:.1f}s. Results: {run_dir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a categorical midpoint generative model."
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--run-dir", default="experiments/categorical_mgm")
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", dest="n_steps", type=int, default=1000_000)
    parser.add_argument(
        "--critic-steps", dest="critic_steps_per_gen", type=int, default=1
    )
    parser.add_argument("--lr-critic", type=float, default=2e-4)
    parser.add_argument("--lr-gen", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--critic-loss", choices=("ce", "mse"), default="ce")
    parser.add_argument("--gen-grad", choices=("stopped", "full"), default="full")
    parser.add_argument(
        "--real-noise", choices=("none", "smooth", "replace"), default="smooth"
    )
    parser.add_argument("--real-noise-eps", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=4.0)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1_000)
    parser.add_argument("--n-samples", type=int, default=12)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
