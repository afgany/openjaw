"""PPO clipped objective implementation (Schulman et al. 2017).

Implements:
  - Clipped surrogate objective
  - Value function loss (MSE)
  - Entropy bonus for exploration
  - Gradient clipping
  - Optional KL-based early stopping

Separated from the trainer for clarity and testability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from openjaw.policy.networks import ActorCritic
from openjaw.training.buffer import RolloutBuffer


@dataclass
class PPOUpdateResult:
    """Results from one PPO update (n_epochs over the buffer)."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float


def compute_ppo_loss(
    policy: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO clipped objective + value loss + entropy bonus.

    Returns:
        (total_loss, info_dict) where info_dict has diagnostic metrics.
    """
    # Current policy distribution and value
    dist = policy.get_action_distribution(obs)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()
    new_values = policy.critic(obs).squeeze(-1)

    # Policy loss (clipped surrogate)
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss (MSE)
    value_loss = nn.functional.mse_loss(new_values, returns)

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    # Diagnostics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean().item()

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return total_loss, info


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
) -> PPOUpdateResult:
    """Run n_epochs of PPO updates over the rollout buffer.

    Returns:
        PPOUpdateResult with mean losses across all updates.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_frac = 0.0
    n_updates = 0

    for epoch in range(n_epochs):
        batches = buffer.get_batches(batch_size)
        for batch in batches:
            loss, info = compute_ppo_loss(
                policy=policy,
                obs=batch["obs"],
                actions=batch["actions"],
                old_log_probs=batch["old_log_probs"],
                advantages=batch["advantages"],
                returns=batch["returns"],
                old_values=batch["old_values"],
                clip_range=clip_range,
                vf_coef=vf_coef,
                ent_coef=ent_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += info["policy_loss"]
            total_value_loss += info["value_loss"]
            total_entropy += info["entropy"]
            total_approx_kl += info["approx_kl"]
            total_clip_frac += info["clip_fraction"]
            n_updates += 1

        # Early stopping on KL divergence
        if target_kl is not None and info["approx_kl"] > 1.5 * target_kl:
            break

    # Explained variance
    values_np = buffer.values[: buffer.pos]
    returns_np = buffer.returns[: buffer.pos]
    var_returns = returns_np.var()
    if var_returns < 1e-8:
        explained_var = 0.0
    else:
        explained_var = float(1.0 - (returns_np - values_np).var() / var_returns)

    return PPOUpdateResult(
        policy_loss=total_policy_loss / max(n_updates, 1),
        value_loss=total_value_loss / max(n_updates, 1),
        entropy=total_entropy / max(n_updates, 1),
        approx_kl=total_approx_kl / max(n_updates, 1),
        clip_fraction=total_clip_frac / max(n_updates, 1),
        explained_variance=explained_var,
    )
