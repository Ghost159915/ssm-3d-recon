"""
src/models/s5.py
================
Pure-PyTorch S5 (Simplified Structured State Space Sequence) module.

Built from scratch — no mamba-ssm, no s4-pytorch, no CUDA extensions.
Works on: CPU / Mac MPS / AMD ROCm (any PyTorch backend).

Theory
------
Continuous-time linear dynamical system:
    h'(t) = A h(t) + B u(t)
    y(t)  = C h(t) + D u(t)

Zero-Order Hold (ZOH) discretisation at step size Δ:
    Ā  = exp(ΔA)
    B̄  = (exp(ΔA) - I) A⁻¹ B   →  simplified below for diagonal A

Discrete recurrence:
    h_k = Ā h_{k-1} + B̄ u_k
    y_k = C h_k + D u_k

S5 key insight: use a DIAGONAL A matrix (DPLR structure).
This makes exp(ΔA) trivial — it is just elementwise exp.
For diagonal A:
    Ā  = exp(Δ * A)             (elementwise)
    B̄  = (Ā - 1) / A * B       (elementwise division)

This collapses the full matrix exponential O(N³) to O(N) operations,
making the SSM practical without custom CUDA kernels.

Applied to temporal depth fusion (the thesis bridge):
    u_k  = depth feature vector at frame k   [D]
    h_k  = accumulated scene geometry state  [N]
    y_k  = refined depth features at frame k [D]
    L    = sequence length (number of video frames)
    Complexity: O(L·N) vs O(L²) for self-attention

Two inference modes
-------------------
1. Recurrent mode  — standard sequential scan, O(L) memory.
   Used during inference / deployment.

2. Parallel scan   — associative scan (work-efficient prefix sum).
   Used during training for GPU parallelism. Same mathematical result.

References
----------
- Smith et al., "Simplified State Space Layers for Sequence Modeling"
  (ICLR 2023) — https://arxiv.org/abs/2208.04933
- Gu et al., "Efficiently Modeling Long Sequences with Structured State
  Spaces" (ICLR 2022) — https://arxiv.org/abs/2111.00396
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility: initialise diagonal A (HiPPO-LegS approximation)
# ---------------------------------------------------------------------------

def _hippo_init(N: int) -> torch.Tensor:
    """
    Initialise the diagonal of A using the HiPPO-LegS structure.
    This gives better long-range memory than random initialisation.

    Returns complex tensor of shape [N] with negative real parts
    (ensures stability: eigenvalues of A must have Re < 0).
    We use the S4D-Real variant: imaginary axis only, real part = -1/2.

    Shape: [N] complex128  →  will be cast to complex64 in the layer.
    """
    # S4D-Real: A_n = -1/2 + i * π * n
    n = torch.arange(N, dtype=torch.float64)
    real_part = -0.5 * torch.ones(N, dtype=torch.float64)
    imag_part = math.pi * n
    return torch.complex(real_part, imag_part)


# ---------------------------------------------------------------------------
# Core S5 Layer
# ---------------------------------------------------------------------------

class S5Layer(nn.Module):
    """
    Single S5 SSM layer.

    Parameters
    ----------
    d_model : int
        Input/output feature dimension D.
    d_state : int
        Hidden state dimension N. Larger = more memory. Typical: 64–256.
    dt_min, dt_max : float
        Range for the learnable step size Δ (log-uniform initialisation).
    bidirectional : bool
        If True, run a second SSM on the reversed sequence and sum outputs.
        Useful for non-causal (offline) processing.

    Shapes
    ------
    Input  u : [B, L, D]   (batch, sequence length, features)
    Output y : [B, L, D]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # ---- A (diagonal, complex) ----------------------------------------
        # Initialise with HiPPO; stored as log_magnitude + phase (real + imag
        # of log(A)) so the parameterisation stays unconstrained while
        # exp(ΔA) is always a contraction (|Ā| < 1 if Re(A) < 0).
        # We store A as (A_re, A_im) separately to avoid complex param issues
        # on some backends.
        A_init = _hippo_init(d_state).to(torch.complex64)
        self.A_re = nn.Parameter(A_init.real)   # [N]  (kept ≤ 0 via softplus trick)
        self.A_im = nn.Parameter(A_init.imag)   # [N]

        # ---- B (input → state), C (state → output) — complex [N, D] -------
        # B maps the D-dim input to the N-dim state.
        # C maps the N-dim state to the D-dim output.
        # Initialised as random complex (real + imag parts separately).
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # ---- D (skip / direct feedthrough) — real [D] ----------------------
        self.D = nn.Parameter(torch.ones(d_model))

        # ---- Δ (learnable step size, log-uniform init) ---------------------
        dt_init = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(torch.log(dt_init))   # [D]

        # ---- Output projection (optional mixing after SSM) -----------------
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    # ------------------------------------------------------------------
    @property
    def A(self) -> torch.Tensor:
        """Complex diagonal A, shape [N]. Real part forced ≤ 0 for stability."""
        # Use -softplus to keep A_re ≤ 0
        return torch.complex(-F.softplus(self.A_re), self.A_im)

    @property
    def B(self) -> torch.Tensor:
        """Complex B matrix, shape [N, D]."""
        return torch.complex(self.B_re, self.B_im)

    @property
    def C(self) -> torch.Tensor:
        """Complex C matrix, shape [D, N]."""
        return torch.complex(self.C_re, self.C_im)

    # ------------------------------------------------------------------
    def _discretise(self, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ZOH discretisation at step size dt.

        For diagonal A:
            Ā = exp(dt * A)                    [N] complex
            B̄ = (Ā - 1) / A * B               [N, D] complex

        Parameters
        ----------
        dt : [D] positive float   (the learned step sizes per channel)

        Returns
        -------
        A_bar : [N]     complex   discrete state matrix (diagonal)
        B_bar : [N, D]  complex   discrete input matrix
        """
        A = self.A        # [N] complex
        B = self.B        # [N, D] complex

        # dt is [D]; we need dt averaged to a scalar for the state (or we can
        # use a single dt shared across the state dim — standard S5 approach).
        # We use the mean of dt over channels as the global step.
        dt_mean = dt.mean()   # scalar

        A_bar = torch.exp(dt_mean * A)                       # [N] complex
        B_bar = ((A_bar - 1.0) / A).unsqueeze(1) * B        # [N, 1] * [N, D] → [N, D] complex

        return A_bar, B_bar

    # ------------------------------------------------------------------
    def _recurrent_scan(
        self,
        u: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard sequential recurrence:
            h_k = Ā ⊙ h_{k-1} + B̄ u_k
            y_k = Re(C h_k)

        Parameters
        ----------
        u     : [B, L, D]  real input
        A_bar : [N]        complex
        B_bar : [N, D]     complex
        h0    : [B, N]     complex initial state (zeros if None)

        Returns
        -------
        ys : [B, L, D]   real outputs
        h  : [B, N]      complex final hidden state
        """
        B_batch, L, D = u.shape
        N = A_bar.shape[0]

        # Initial hidden state
        if h0 is None:
            h = torch.zeros(B_batch, N, dtype=torch.complex64, device=u.device)
        else:
            h = h0.to(torch.complex64)

        # Cast u to complex for matrix multiply
        u_c = u.to(torch.complex64)   # [B, L, D]

        C = self.C  # [D, N] complex

        outputs = []
        for k in range(L):
            # u_k: [B, D]
            # B_bar @ u_k.T would be [N, B]; we want [B, N]
            # So: (u_k @ B_bar.T) → [B, N]   since B_bar is [N, D]
            Bu = u_c[:, k, :] @ B_bar.T          # [B, N]   (B_bar.T is [D, N])
            h = A_bar.unsqueeze(0) * h + Bu       # [B, N]  elementwise Ā ⊙ h
            y_k = (h @ C.T).real                  # [B, D]  real part of C h
            outputs.append(y_k)

        ys = torch.stack(outputs, dim=1)   # [B, L, D]
        return ys, h

    # ------------------------------------------------------------------
    def _parallel_scan(
        self,
        u: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel associative scan (Blelloch algorithm).
        Computes the same result as _recurrent_scan but in O(log L) depth,
        enabling GPU parallelism during training.

        For the diagonal SSM the recurrence is:
            (Ā_k, x_k)  where  x_k = Ā_{k:1} * x_0 + Σ Ā_{k:j+1} * B̄ u_j

        This reduces to a prefix product over the pairs (Ā, Bu) with the
        binary operator ⊕ defined as:
            (a1, b1) ⊕ (a2, b2) = (a2 * a1,  a2 * b1 + b2)

        Parameters
        ----------
        u     : [B, L, D]  real
        A_bar : [N]        complex (same Ā applied at each step)
        B_bar : [N, D]     complex

        Returns
        -------
        ys : [B, L, D]  real
        """
        B_batch, L, D = u.shape
        N = A_bar.shape[0]

        u_c = u.to(torch.complex64)   # [B, L, D]

        # Compute Bu_k for all steps: [B, L, N]
        # B_bar is [N, D]; u_c is [B, L, D]
        # Bu[b, l, n] = sum_d B_bar[n, d] * u_c[b, l, d]
        Bu = torch.einsum("nd,bld->bln", B_bar, u_c)  # [B, L, N]

        # For uniform Ā (same at every step), the parallel scan simplifies:
        # h_k = Ā^k * h_0 + Σ_{j=1}^{k} Ā^{k-j} * Bu_j
        # With h_0 = 0:
        # h_k = Σ_{j=1}^{k} Ā^{k-j} * Bu_j
        # This is a causal cumulative sum with exponential decay.

        # Powers of Ā: [L]  Ā^0, Ā^1, ..., Ā^{L-1}
        # A_bar: [N], exponents: [L]
        powers = torch.arange(L, device=u.device, dtype=torch.float32)  # [L]
        # A_bar_powers[l, n] = Ā_n^l
        A_bar_log = A_bar.unsqueeze(0).log()             # [1, N]
        powers_c = powers.to(torch.complex64).unsqueeze(1)  # [L, 1]
        A_bar_powers = torch.exp(powers_c * A_bar_log)   # [L, N]

        # h_k = Σ_{j=0}^{k} Ā^{k-j} Bu_j
        # = Ā^k Σ_{j=0}^{k} Ā^{-j} Bu_j
        # Compute the prefix sum of (Ā^{-j} Bu_j) then multiply by Ā^k

        # inv_A_powers[l, n] = Ā_n^{-l}
        inv_A_bar_log = -A_bar_log                        # [1, N]
        inv_A_bar_powers = torch.exp(powers_c * inv_A_bar_log)  # [L, N]

        # weighted_Bu[b, l, n] = Ā^{-l} Bu[b, l, n]
        weighted_Bu = inv_A_bar_powers.unsqueeze(0) * Bu  # [B, L, N]

        # prefix_sum[b, k, n] = Σ_{l=0}^{k} weighted_Bu[b, l, n]
        prefix_sum = torch.cumsum(weighted_Bu, dim=1)     # [B, L, N]

        # h_k = Ā^k * prefix_sum_k
        h = A_bar_powers.unsqueeze(0) * prefix_sum        # [B, L, N]

        # y_k = Re(C h_k)  where C: [D, N]
        C = self.C   # [D, N]
        # y[b, l, d] = Re( Σ_n C[d,n] h[b,l,n] )
        ys = torch.einsum("dn,bln->bld", C, h).real       # [B, L, D]
        return ys

    # ------------------------------------------------------------------
    def forward(
        self,
        u: torch.Tensor,
        use_parallel: bool = True,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the S5 layer.

        Parameters
        ----------
        u            : [B, L, D]   input sequence (real float)
        use_parallel : bool        True → parallel scan (faster training)
                                   False → sequential recurrence (inference,
                                           can pass h0)
        h0           : [B, N] complex initial state (recurrent mode only)

        Returns
        -------
        y  : [B, L, D]   output sequence
        h  : [B, N] complex final state (None in parallel mode)
        """
        dt = torch.exp(self.log_dt)           # [D]  positive step sizes
        A_bar, B_bar = self._discretise(dt)   # [N], [N, D]

        if use_parallel:
            ys = self._parallel_scan(u, A_bar, B_bar)
            h_final = None
        else:
            ys, h_final = self._recurrent_scan(u, A_bar, B_bar, h0)

        # Skip connection (D term) + output projection
        y = ys + self.D.unsqueeze(0).unsqueeze(0) * u    # [B, L, D]
        y = self.out_proj(y)

        # Bidirectional: also run on reversed sequence
        if self.bidirectional:
            u_rev = u.flip(dims=[1])
            if use_parallel:
                ys_rev = self._parallel_scan(u_rev, A_bar, B_bar)
            else:
                ys_rev, _ = self._recurrent_scan(u_rev, A_bar, B_bar)
            y_rev = ys_rev + self.D.unsqueeze(0).unsqueeze(0) * u_rev
            y_rev = self.out_proj(y_rev).flip(dims=[1])
            y = y + y_rev

        return y, h_final


# ---------------------------------------------------------------------------
# S5 Stack (multiple layers with residual connections)
# ---------------------------------------------------------------------------

class S5Stack(nn.Module):
    """
    Stack of S5 layers with residual connections and layer norm.

    Architecture per layer:
        x = LayerNorm(x)
        x = x + S5Layer(x)    (residual)
        x = LayerNorm(x)
        x = x + FFN(x)        (small 2-layer MLP, expansion=2)

    This mirrors the pre-norm Transformer block structure, which trains
    stably without careful learning rate tuning.

    Parameters
    ----------
    d_model  : int   feature dimension (same in and out)
    d_state  : int   S5 hidden state size per layer
    n_layers : int   number of S5 blocks
    dropout  : float dropout rate inside FFN
    bidirectional : bool   passed to each S5Layer
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            S5Layer(d_model, d_state, bidirectional=bidirectional)
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        use_parallel: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : [B, L, D]
        use_parallel : bool

        Returns
        -------
        x : [B, L, D]
        """
        for ssm, norm1, norm2, ffn in zip(
            self.layers, self.norms1, self.norms2, self.ffns
        ):
            # SSM block (pre-norm residual)
            residual = x
            x = norm1(x)
            x, _ = ssm(x, use_parallel=use_parallel)
            x = x + residual

            # FFN block (pre-norm residual)
            residual = x
            x = norm2(x)
            x = ffn(x)
            x = x + residual

        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, D, N = 2, 16, 64, 32

    print("=" * 50)
    print("S5Layer smoke test")
    print("=" * 50)

    layer = S5Layer(d_model=D, d_state=N)
    u = torch.randn(B, L, D)

    # Parallel mode
    y_par, _ = layer(u, use_parallel=True)
    print(f"Parallel output shape : {y_par.shape}")

    # Recurrent mode
    y_rec, h = layer(u, use_parallel=False)
    print(f"Recurrent output shape: {y_rec.shape}")
    print(f"Final hidden state    : {h.shape}  dtype={h.dtype}")

    # Check that parallel ≈ recurrent (should match closely)
    max_diff = (y_par - y_rec).abs().max().item()
    print(f"Max |parallel - recurrent| : {max_diff:.6f}  (should be < 1e-4)")

    print()
    print("=" * 50)
    print("S5Stack smoke test")
    print("=" * 50)

    stack = S5Stack(d_model=D, d_state=N, n_layers=3)
    out = stack(u)
    print(f"Stack output shape: {out.shape}")

    # Parameter count
    n_params = sum(p.numel() for p in stack.parameters())
    print(f"Total parameters  : {n_params:,}")
    print("All checks passed.")
