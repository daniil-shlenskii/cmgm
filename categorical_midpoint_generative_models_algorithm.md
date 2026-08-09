# Categorical Midpoint Generative Models

This note specifies a simple training algorithm for **Categorical Midpoint Generative Models (C-MGM)** on fixed-length token sequences. It uses:

- tokenwise one-hot representations;
- optional noising of real data;
- a simplex relaxation of generated tokens;
- a symmetric categorical mixer;
- either an MSE critic or two cross-entropy denoisers;
- the original MGM generator payoff.

The main algorithm is a relaxed, differentiable MGM game on the token simplex. A lower-variance, biased generator update is included as an optional variant.

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| \(L\) | Sequence length |
| \(V\) | Vocabulary size, including EOS/PAD if needed |
| \(x^+\sim p_{\mathrm{data}}\) | Real token sequence |
| \(R=\operatorname{OH}(x^+)\in\{0,1\}^{L\times V}\) | Real sequence in one-hot form |
| \(z\sim p_z\) | Generator noise or latent input |
| \(g_\theta(z)\in\mathbb R^{L\times V}\) | Generator logits |
| \(P_\theta(z)\in\Delta^{L\times V}\) | Relaxed generated sequence |
| \(t\in[0,1/2]\) | Interpolation time |
| \(B\sim\operatorname{Bernoulli}(1/2)\) | Hidden global MGM flip |
| \(\widetilde Y_t\) | Symmetric mixed observation given to the critic |
| \(\Delta=R_\varepsilon-P_\theta\) | Relaxed endpoint displacement |

Here \(\Delta^V\) denotes the vocabulary simplex. All norms and inner products below sum over token positions and vocabulary coordinates and are averaged over the batch. In practice, divide sequence losses by \(L\).

## 2. One-hot data and optional real-data noising

The basic real endpoint is

\[
R=\operatorname{OH}(x^+).
\]

Optionally move it slightly into the simplex interior:

\[
R_\varepsilon=(1-\varepsilon)R+\varepsilon U,
\qquad
U_{i,v}=\frac1V,
\qquad 0\leq\varepsilon<1.
\]

Use \(R_\varepsilon=R\) when \(\varepsilon=0\). For \(\varepsilon<1\), argmax decoding is unchanged:

\[
\arg\max_v R_{\varepsilon,i,v}=x_i^+.
\]

The noised endpoint avoids comparing finite-temperature generator probabilities only against exact simplex vertices.

## 3. Relaxed generator endpoint

The generator outputs token logits and converts them to simplex values:

\[
P_\theta(z)
=
\operatorname{softmax}\!\left(\frac{g_\theta(z)}{T_G}\right),
\qquad T_G>0.
\]

No categorical token is sampled during training. The latent \(z\) supplies sample-level stochasticity, and shared latent information must be available to all sequence positions.

The relaxed displacement is

\[
\Delta=R_\varepsilon-P_\theta(z).
\]

## 4. Symmetric categorical mixer

Choose a monotone schedule \(\kappa:[0,1]\to[0,1]\) satisfying

\[
\kappa(0)=0,
\qquad
\kappa(1)=1,
\qquad
\kappa(1-t)=1-\kappa(t).
\]

The simplest choice is \(\kappa(t)=t\).

Sample

\[
t\sim p_t\text{ on }[0,1/2],
\qquad
B\sim\operatorname{Bernoulli}(1/2).
\]

Define the probability that a position comes from the real endpoint:

\[
a_{t,B}=
\begin{cases}
\kappa(t), & B=0,\\
1-\kappa(t), & B=1.
\end{cases}
\]

The flip \(B\) must not be given to the critic.

### Default: independent token mixer

For each position, sample

\[
M_i\sim\operatorname{Bernoulli}(a_{t,B})
\]

independently, and set

\[
\widetilde Y_{t,i}
=
M_iR_{\varepsilon,i}+(1-M_i)P_{\theta,i}.
\]

The mask \(M\) is independent of \(\theta\), so no discrete-gradient estimator is required. Do not reveal \(M\) to the critic.

### Alternative: fixed-count mixer

Set \(K=\lfloor La_{t,B}\rfloor\), sample exactly \(K\) positions uniformly, and use the real endpoint at those positions. This removes variation in the number of real tokens. If this option is used, sampling \(K\) directly is often clearer than treating the path as continuous in \(t\).

### Optional symmetric observation noise

After mixing, one may apply

\[
\widetilde Y_t
\leftarrow
(1-\rho_t)\widetilde Y_t+\rho_t U,
\]

where

\[
\rho_0=\rho_1=0,
\qquad
\rho_t=\rho_{1-t}.
\]

Use the same observation construction for critic and generator updates.

## 5. Critic option A: direct MSE field

Let

\[
f_\psi:\Delta^{L\times V}\times[0,1/2]
\to\mathbb R^{L\times V}
\]

be an unconstrained Transformer field. For the critic update, stop gradients through the generator endpoint and train

\[
\mathcal L_{\mathrm{critic}}^{\mathrm{MSE}}
=
\mathbb E
\left[
\left\|
f_\psi(\widetilde Y_t,t)
-(R_\varepsilon-P_\theta)
\right\|_2^2
\right].
\]

Its population optimum is

\[
f^*(y,t)
=
\mathbb E[R_\varepsilon-P_\theta\mid \widetilde Y_t=y,t].
\]

### MSE critic step

1. Sample \(x^+,z,t,B,M\).
2. Compute \(R_\varepsilon\).
3. Compute \(P_\theta\) and detach it.
4. Construct \(\widetilde Y_t\) from the detached endpoint.
5. Minimize \(\mathcal L_{\mathrm{critic}}^{\mathrm{MSE}}\) over \(\psi\).

## 6. Critic option B: two CE denoisers

Let the critic produce two categorical endpoint denoisers:

\[
q_{0,\psi}(\cdot\mid\widetilde Y_t,t),
\qquad
q_{1,\psi}(\cdot\mid\widetilde Y_t,t),
\]

where \(q_0\) predicts the generated endpoint and \(q_1\) predicts the real endpoint. A practical implementation uses one shared Transformer and two endpoint-conditioned nonlinear heads.

Use soft-label cross-entropy:

\[
\mathcal L_{\mathrm{critic}}^{\mathrm{CE}}
=
-\mathbb E
\left[
\langle P_\theta,\log q_{0,\psi}(\widetilde Y_t,t)\rangle
+
\langle R_\varepsilon,\log q_{1,\psi}(\widetilde Y_t,t)\rangle
\right].
\]

Stop gradients through \(P_\theta\) and \(\widetilde Y_t\) during the critic update. At population optimum,

\[
q_0^*(y,t)=\mathbb E[P_\theta\mid \widetilde Y_t=y,t],
\]

\[
q_1^*(y,t)=\mathbb E[R_\varepsilon\mid \widetilde Y_t=y,t].
\]

Define the field used by the generator as

\[
f_\psi(y,t)=q_{1,\psi}(y,t)-q_{0,\psi}(y,t).
\]

Then the population-optimal CE field equals the population-optimal MSE field:

\[
f^*(y,t)
=
\mathbb E[R_\varepsilon-P_\theta\mid\widetilde Y_t=y,t].
\]

### CE critic step

1. Sample \(x^+,z,t,B,M\).
2. Compute \(R_\varepsilon\).
3. Compute \(P_\theta\) and detach it.
4. Construct \(\widetilde Y_t\) from the detached endpoint.
5. Predict \(q_0,q_1\).
6. Minimize \(\mathcal L_{\mathrm{critic}}^{\mathrm{CE}}\) over \(\psi\).
7. During generator training, use \(f_\psi=q_{1,\psi}-q_{0,\psi}\).

Do not minimize the CE value with respect to the generator. CE is only a surrogate for estimating the optimal midpoint field.

## 7. Generator objective

For either critic option, freeze the critic parameters and minimize the MGM payoff

\[
\boxed{
\mathcal L_{\mathrm{generator}}
=
\mathbb E
\left[
2\left\langle
f_\psi(\widetilde Y_t,t),
R_\varepsilon-P_\theta
\right\rangle
-
\left\|f_\psi(\widetilde Y_t,t)\right\|_2^2
\right].
}
\]

The generator minimizes this expression; the critic parameters are not updated during this step.

### Option G1: full relaxed gradient

Recompute \(P_\theta\) without detaching it and construct \(\widetilde Y_t\) from this endpoint. Backpropagate through both paths:

\[
P_\theta\to R_\varepsilon-P_\theta,
\]

\[
P_\theta\to\widetilde Y_t\to f_\psi(\widetilde Y_t,t).
\]

This is the exact pathwise gradient of the relaxed MGM objective.

### Option G2: stopped-observation relaxation

For a lower-variance but biased update, construct

\[
\widetilde Y_t
=
\operatorname{Mixer}
\bigl(\operatorname{sg}(P_\theta),R_\varepsilon,t,B,M\bigr)
\]

and detach the resulting field:

\[
\bar f=\operatorname{sg}(f_\psi(\widetilde Y_t,t)).
\]

Only the displacement path remains differentiable. Up to generator-independent terms, the loss becomes

\[
\boxed{
\mathcal L_{\mathrm{generator}}^{\mathrm{SG}}
=
-2\,\mathbb E\langle P_\theta,\bar f\rangle.
}
\]

This semi-gradient is often more stable, but it is not the exact gradient of the relaxed midpoint divergence. It preserves the desired fixed point because \(f^*=0\) when the generated and real endpoint distributions match.

## 8. Complete alternating algorithm

```text
Inputs:
    generator G_theta
    critic C_psi
    critic_type in {MSE, CE}
    generator_gradient in {FULL, STOPPED}
    time distribution p_t on [0, 1/2]
    symmetric schedule kappa
    generator temperature T_G
    real-data noise epsilon

repeat:
    # ----------------------------------------------------------
    # 1. Critic update
    # ----------------------------------------------------------
    sample real tokens x_plus ~ p_data
    sample latent z ~ p_z
    sample t ~ p_t
    sample hidden flip B ~ Bernoulli(1/2)

    R = one_hot(x_plus)
    R_eps = (1 - epsilon) * R + epsilon * UniformSimplex
    P = stop_gradient(softmax(G_theta(z) / T_G))

    a = kappa(t) if B == 0 else 1 - kappa(t)
    sample M_i ~ Bernoulli(a) independently
    Y_i = M_i * R_eps_i + (1 - M_i) * P_i

    if critic_type == MSE:
        f = C_psi(Y, t)
        critic_loss = mean_squared_error(f, R_eps - P)
    else:
        q0, q1 = C_psi(Y, t)
        critic_loss = soft_CE(P, q0) + soft_CE(R_eps, q1)

    update psi to minimize critic_loss

    # ----------------------------------------------------------
    # 2. Generator update using a fresh batch
    # ----------------------------------------------------------
    sample fresh x_plus, z, t, B, M
    compute R_eps as above
    P = softmax(G_theta(z) / T_G)

    if generator_gradient == FULL:
        Y = Mixer(P, R_eps, t, B, M)
        f = C_psi(Y, t)                  # MSE critic
        # or f = q1(Y,t) - q0(Y,t)       # CE critic
        generator_loss = 2 * inner(f, R_eps - P) - squared_norm(f)
    else:
        Y = Mixer(stop_gradient(P), R_eps, t, B, M)
        f = stop_gradient(C_psi(Y, t))   # MSE critic
        # or stop_gradient(q1 - q0)      # CE critic
        generator_loss = -2 * inner(P, f)

    update theta to minimize generator_loss

until convergence
```

Use fresh samples for the critic and generator steps. The flip \(B\) and mask \(M\) are used to construct the observation but are never passed to the critic.

## 9. Optional denoising warm-up from real data

Before MGM training, compatible generator and critic weights may be initialized using real-data denoising. This is only a weight warm-up; it is not a standalone one-step generative objective.

1. Sample a real sequence \(R\).
2. Sample random one-hot noise \(N\), for example iid uniform vocabulary tokens.
3. Sample a corruption level \(s\in[0,s_{\max}]\) and mask \(C_i\sim\operatorname{Bernoulli}(s)\). A value \(s_{\max}<1\) avoids training only on fully uninformative inputs.
4. Construct

   \[
   Z_{s,i}=(1-C_i)R_i+C_iN_i.
   \]

5. Train a denoiser \(D_\eta(Z_s,s)\) with tokenwise CE against \(R\).
6. Copy compatible denoiser weights into the generator and critic trunk, then continue with the MGM objective.

This warm-up is optional. It uses only real data and does not require a pretrained diffusion teacher. It should not be interpreted as proving that \(D_\eta(N,1)\) is already a good one-step generator: under complete corruption, the CE-optimal denoiser may ignore \(N\) and output token marginals.

## 10. Sampling

At inference:

1. Sample \(z\sim p_z\).
2. Compute

   \[
   P_\theta(z)=\operatorname{softmax}(g_\theta(z)/T_G).
   \]

3. Decode all positions in parallel:

   \[
   \widehat x_i=\arg\max_v P_{\theta,i,v}.
   \]

Categorical sampling from \(P_\theta\) is also possible, but argmax avoids adding independent token noise after the shared latent has already selected the sequence-level mode.

## 11. Correctness summary

For a fixed generator distribution and an expressive critic:

- the MSE critic recovers

  \[
  \mathbb E[R_\varepsilon-P_\theta\mid\widetilde Y_t,t];
  \]

- the two-CE critic recovers the same field through \(q_1^*-q_0^*\);
- inserting this field into the generator payoff gives the relaxed midpoint divergence;
- with a symmetric, endpoint-revealing mixer, the relaxed divergence vanishes exactly when the relaxed endpoint distributions match;
- the full generator update differentiates this relaxed divergence;
- the stopped-observation update is a practical biased semi-gradient with the same desired zero-field fixed point.

## 12. Minimal recommended baseline

For the first implementation, use:

- tokenwise one-hot data with small \(\varepsilon\)-smoothing;
- a shared-latent parallel Transformer generator;
- independent token mixing with \(\kappa(t)=t\);
- a shared Transformer critic with two endpoint-conditioned CE heads;
- the stopped-observation generator update for initial stability;
- the full relaxed gradient as the main ablation;
- loss normalization by sequence length;
- monitoring of token entropy and \(1-\max_vP_{\theta,i,v}\).
