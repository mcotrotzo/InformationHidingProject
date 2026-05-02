"""Differentiable simulator of steganography for grayscale.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import math
from time import time
from scipy.optimize import root_scalar
import torch
torch._dynamo.config.cache_size_limit = 16  # allowed versions per compiled function
torch._functorch.config.donated_buffer = False  # disable donated buffers
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Tuple, Dict, Callable, Optional
import warnings

TIME_SPENT = []

# ========== ADJUSTING ==========
@torch.compile
def adjust_rho(rho: torch.Tensor, x0: torch.Tensor, wet_cost: float = 10**10) -> torch.Tensor:
    """adjusts rho according to the saturated areas in x0.

    rho has a shape (B, 3, H, W), where the channels carry costs ordered as (rho_0, rho_p1, rho_m1).
    x0 has a shape (B, 1, H, W)
    """
    # replace inf/nan/>wet
    rho = torch.where(
        torch.isfinite(rho) & (rho < wet_cost),
        rho,
        torch.tensor(wet_cost, device=rho.device, dtype=rho.dtype),
    )  # B 3 H W

    # boundaries (1/255 = .00392)
    mask = torch.zeros(rho.shape, dtype=torch.bool, device=rho.device)
    mask[:, 1:2] = (x0 >= .999)  # B 1 H W
    mask[:, 2:3] = (x0 <= .001)  # B 1 H W
    # masking
    rho = torch.where(mask, torch.tensor(wet_cost, device=rho.device), rho)  # B 3 H W
    return rho  # B 27 H W


# ========== ENTROPY CONSTRAINT ==========
@torch.compile
def entropy(p: torch.Tensor, dim=None) -> torch.Tensor:
    assert p.size(1) in {2, 3, 6, 9, 27}, 'strange q-arity encountered'
    #
    eps = torch.finfo(p.dtype).eps
    p = torch.clamp(p, eps, 1-eps)  # B 27 H W
    #
    if dim is None:
        dim = tuple(range(1, p.ndim))
    h_hat = -torch.sum(p * torch.log2(p), dim=dim)
    return h_hat


def exponential_lambda_search(rho, m, objective):
    B = rho.shape[0]
    device = rho.device
    lbd_l = torch.full((B,), 1e-30, dtype=rho.dtype, device=device)
    lbd_r = torch.full((B,), 1e-30, dtype=rho.dtype, device=device)
    active_mask = torch.ones(B, dtype=torch.bool, device=device)
    for _ in range(50): # 1e-30 to 1e16 in 10x steps is ~46 iterations
        if not active_mask.any():
            break
        _, v = objective(rho, lbd_r.view(-1, 1, 1, 1))  # current payload for active images
        still_too_high = (v >= m) & (lbd_r < 1e16)  # check which images remain active
        lbd_l = torch.where(still_too_high, lbd_r, lbd_l)  # shift lower bound
        lbd_r = torch.where(still_too_high, lbd_r * 10, lbd_r)  # multiply upper bound
        active_mask = still_too_high  # carry the active images
    return lbd_l, lbd_r


def calc_lambda(rho, m, objective, xtol=1e-3, max_iter=100):
    print("Using binary search for lambda.")
    """Lambda search via binary search, to be called per batch."""
    device = rho.device
    B = rho.shape[0]
    if isinstance(m, (int, float)):
        m = torch.full((B,), m, dtype=rho.dtype, device=device)
    # exponential search
    lbd_l, lbd_r = exponential_lambda_search(rho, m, objective)
    # binary search
    for _ in range(max_iter):  # log2((max-min)/xtol) steps needed (70 is enough for f32 and 1e-3)
        mid = (lbd_l + lbd_r) / 2  # middle
        _, v = objective(rho, mid.view(-1, 1, 1, 1))  # current payload
        too_high = v > m  # splitting rule
        lbd_l = torch.where(too_high, mid, lbd_l)  # lower half
        lbd_r = torch.where(~too_high, mid, lbd_r)  # upper half
        if (lbd_r - lbd_l).max() < xtol:  # early exit
            break
    #
    lbd = (lbd_l + lbd_r) / 2  # midpoint is the final lambda
    _, h_hat = objective(rho, lbd.view(-1, 1, 1, 1))  # final entropy
    return lbd, h_hat


CUSTOM_LAMBDA = calc_lambda

@torch.compile
def average_payload(
    rho: torch.Tensor = None,
    lbda: float = None,
) -> Tuple[torch.Tensor, float]:
    """Objective: code-native entropy.

    rho is a cost tensor of shape (B, C, H, W)
    lbda is a float parameter
    """
    p = F.softmax(-lbda * rho, dim=1)  # get selection channel
    h_hat = entropy(p)
    return p, h_hat


class find_lambda(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, rho, m, n, objective: Callable = average_payload):
        B = rho.size(0)
        lbdas = []
        t0 = time()
        lbdas, h_hat = CUSTOM_LAMBDA(
            rho=rho,
            m=m,
            objective=objective,
        )
        t1 = time()
        global TIME_SPENT
        TIME_SPENT.append(t1 - t0)
        lbdas = lbdas.view(B, 1, 1, 1)
        with torch.enable_grad():
            r = rho.detach().requires_grad_(True)
            l = lbdas.detach().requires_grad_(True)

        ctx.save_for_backward(r, l)
        ctx.objective = objective
        # if (abs(m / n - h_hat / n) > 1e-3).any():
        #     warnings.warn(f'calc_lambda diverged, {h_hat.item() / n}')
        return lbdas.detach().clone().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            rho, lbda = ctx.saved_tensors  # 1 9 H W
            _, h_hat = ctx.objective(rho, lbda)

            # Gradients
            grad_h_lbda = torch.autograd.grad(h_hat, lbda, torch.ones_like(h_hat), retain_graph=True)[0]
            grad_h_rho = torch.autograd.grad(h_hat, rho, torch.ones_like(h_hat), retain_graph=False)[0]

            # Implicit gradient
            g = -grad_h_rho / (grad_h_lbda - 1e-9)
        return grad_output * g, None, None, None


# ========== SIMULATION ==========
@torch.compile
def _simulate(
    p: torch.Tensor,
    rand: torch.Tensor,  # B ~ U(0,1), B 1 H W
    simulator_method: str = 'hard',
) -> torch.Tensor:
    """Simulates ternary steganographic embedding via MI according to the distributions p."""
    B, C, H, W = p.size()
    p_cum = torch.cumsum(p, dim=1)
    #
    indices = torch.argmax((rand < p_cum).to(torch.int8), dim=1)  # B H W
    delta3_hard = F.one_hot(indices, num_classes=C).permute(0, 3, 1, 2).to(p.dtype)
    delta3_hard = torch.round(delta3_hard)
    if simulator_method == 'hard':
        delta3 = delta3_hard
    elif simulator_method == 'probability-ste':
        delta3 = (delta3_hard - p).detach() + p  # project p to p3
    else:
        raise NotImplementedError(f'unknown simulator method {simulator_method}')
    return delta3

@torch.compile
def _simulate_differentiable(
    p: torch.Tensor,  # B 3 H W, per-pixel categorical probabilities
    rand: torch.Tensor,  # B ~ U(0,1), B 3 H W
    tau: float = 1.0,  # temperature for Gumbel-Softmax (1.0 is default)
    simulator_method: str = 'gumbel-softmax',
) -> torch.Tensor:
    """
    Differentiable q-ary sampling with Straight-Through Estimator (hard Gumbel-Softmax).

    Forward pass: hard categorical selection -> real delta pattern
    Backward pass: gradient flows through soft expectation
    """
    # Gumbel noise for sampling
    eps = 1e-20
    gumbel_noise = -torch.log(-torch.log(rand + eps) + eps)

    # Soft sample
    logits = torch.log(p + eps)  # avoid log(0)
    delta_soft = torch.softmax((logits + gumbel_noise) / tau, dim=1)  # B 3 H W

    # Hard categorical sample (forward)
    idx = delta_soft.argmax(dim=1, keepdim=True)  # B 1 H W
    delta_hard = torch.zeros_like(p).scatter_(1, idx, 1.0)

    # Straight-through: forward = hard, backward = soft
    if simulator_method == 'gumbel-softmax':
        delta3 = (delta_hard - delta_soft).detach() + delta_soft  # B 3 H W
    elif simulator_method == 'vanilla':
        delta3 = delta_soft
    else:
        raise NotImplementedError(f'unknown simulator method {simulator_method}')
    return delta3

def generate_randomness(shape: Tuple[int], stego_seed: int, device: torch.device):
    """Generates randomness of a shape with the seed on the device."""
    # reproducibility
    rng = torch.Generator(device=device)
    if stego_seed is not None:
        rng.manual_seed(stego_seed)
    # randomness
    rand = torch.rand(shape, generator=rng, device=device)
    return rand

def simulate(
    p: torch.Tensor,          # B 3 H W, per-pixel categorical probabilities
    tau: float = 1.0,         # temperature for Gumbel-Softmax (1.0 is default)
    one_hot: bool = False,
    simulator_method: str = 'hard',  # expectation, gumbel-softmax, vanilla, hard-expectation, hard
    stego_seed: int = None,
) -> torch.Tensor:
    """
    Differentiable q-ary sampling with Straight-Through Estimator (hard Gumbel-Softmax).

    Forward pass: hard categorical selection -> real delta pattern
    Backward pass: gradient flows through soft expectation
    """
    B, C, H, W = p.shape
    # --------------------------------------
    # # simulate
    if simulator_method in {'hard', 'hard-expectation'}:
        rand = generate_randomness((B, 1, H, W), stego_seed=stego_seed, device=p.device)
        delta = _simulate(
            p=p,
            rand=rand,
            simulator_method=simulator_method,
        )  # B 3 H W
    elif simulator_method in {'gumbel-softmax', 'vanilla'}:
        rand = generate_randomness((B, C, H, W), stego_seed=stego_seed, device=p.device)
        delta = _simulate_differentiable(
            p=p,
            rand=rand,
            tau=tau,
            simulator_method=simulator_method,
        )  # B 3 H W
    else:
        delta = p  # B 3 H W

    # --------------------------------------
    # Map categorical index to delta pattern
    if not one_hot:
        H = torch.tensor([[0, +1, -1]]).to(torch.float32).to(p.device)  # 1 3
        delta = torch.einsum('bchw, qc -> bqhw', delta, H)  # B 1 H W
    return delta


def simulateExperiment(customLambda: Optional[Callable] = None):
    global CUSTOM_LAMBDA
    global TIME_SPENT
    TIME_SPENT = []

    if customLambda is not None:
        CUSTOM_LAMBDA = customLambda
    
    import conseal as cl
    import numpy as np
    from PIL import Image
    import sealwatch as sw

    # calculate cost
    x = np.array(Image.open('seal6.png'))
    rho = cl.hill._costmap.compute_cost(x)
    rho = np.stack([np.zeros_like(rho), rho, rho], axis=0)  # symmetric costs

    # convert to torch
    device = torch.device('cpu')
    x0 = torch.from_numpy(x[None, None] / 255.).to(torch.float32).to(device)
    rho = torch.from_numpy(rho[None]).to(torch.float32).to(device)
    rho = adjust_rho(rho=rho, x0=x0, wet_cost=10**10)

    # stego parameters
    B, C, H, W = x0.size()
    N = C * H * W  # number of elements
    alpha = .4  # bpp

    # simulated embedding
    lbda = find_lambda.apply(rho, alpha * N, N)
    p, _ = average_payload(rho, lbda)
    delta = simulate(p=p, one_hot=False, simulator_method='gumbel-softmax', stego_seed=12345)

    # stego parameters
    h = entropy(p)
    alpha_hat = h / N
    beta = torch.sum(p[:, 1:], dim=(1, 2, 3)) / N
    beta_hat = torch.mean(torch.abs(delta), dim=(1, 2, 3))
    return alpha_hat,beta,beta_hat,TIME_SPENT

if __name__ == '__main__':
    simulateExperiment()
