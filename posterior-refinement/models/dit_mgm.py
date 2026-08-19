import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit_fmlm_plus import DIT_FMLM_PLUS


class DIT_MGM_GENERATOR(DIT_FMLM_PLUS):
    """G_theta(z, puzzle) -> per-position vocab logits, one forward pass."""

    def __init__(self, config, vocab_size: int, noise_dim: int):
        super().__init__(config, vocab_size)
        cond_dim = config.model.cond_dim
        self.z_proj = nn.Linear(noise_dim, cond_dim)

    def forward(self, x, z, seg_ids=None):
        h = self.vocab_embed(x)
        if self.cond_seg_embed is not None and seg_ids is not None:
            h = h + self.cond_seg_embed(seg_ids)

        t_cond = F.silu(self.z_proj(z))  # (B, cond_dim) global conditioning

        rotary_cos_sin = self.rotary_emb(h)
        with torch.amp.autocast(device_type=h.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                h = self.blocks[i](h, rotary_cos_sin, c=t_cond, seqlens=None)
            h = self.output_layer(h, c=t_cond)
        return h
