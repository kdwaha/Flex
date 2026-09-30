"""pFLAlign optimizer from Algorithm 1 of arXiv:2605.02143."""

from __future__ import annotations

import torch


def canonical_peft_name(name: str) -> str:
    return name.replace(".default", "")


class PFLAlignOptimizer(torch.optim.Optimizer):
    """Coordinate-wise pFLAlign update with persistent client ``v`` and ``P``."""

    def __init__(self, named_parameters, lr, beta, epsilon, local_steps, delta, v, preconditioner, variant="full"):
        named_parameters = [(name, param) for name, param in named_parameters if param.requires_grad]
        super().__init__([param for _, param in named_parameters], {"lr": lr})
        self.names = {id(param): canonical_peft_name(name) for name, param in named_parameters}
        self.beta = beta
        self.epsilon = epsilon
        self.local_steps = max(1, int(local_steps))
        self.delta = delta
        self.variant = variant

        for _, param in named_parameters:
            name = self.names[id(param)]
            state = self.state[param]
            # Algorithm 1 resets m every communication round, while v and P persist.
            state["m"] = torch.zeros_like(param)
            state["v"] = v[name].to(device=param.device, dtype=param.dtype).clone()
            state["P"] = preconditioner[name].to(device=param.device, dtype=param.dtype).clone()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        beta = self.beta
        one_minus_beta = 1.0 - beta
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                name = self.names[id(param)]
                state = self.state[param]
                m, v, preconditioner = state["m"], state["v"], state["P"]
                m.mul_(beta).add_(grad, alpha=one_minus_beta)
                v.mul_(beta).addcmul_(grad, grad, value=one_minus_beta)
                alpha = 1.0 - one_minus_beta * grad.square() / (v + self.epsilon)
                preconditioner.mul_(alpha).add_(
                    one_minus_beta * m.square() / (v + self.epsilon)
                )
                variance = (v - m.square()).clamp_min_(0.0)
                z = m.abs() / torch.sqrt(2.0 * variance + self.epsilon)
                delta = self.delta[name].to(device=param.device, dtype=param.dtype)
                gamma = 0.5 - 0.5 * torch.erf(z) * torch.sign(-m * delta)
                if self.variant in {"no_correction", "no_personalization", "sgd"}:
                    gamma = torch.zeros_like(gamma)
                elif self.variant == "constant_gamma":
                    gamma = torch.full_like(gamma, 0.5)
                elif self.variant == "constant_gamma_one":
                    # Remove exactly the initial Delta over T optimizer steps.
                    gamma = torch.ones_like(gamma)
                elif self.variant == "hard_gamma":
                    gamma = 0.5 + 0.5 * torch.sign(m * delta)
                direction = grad if self.variant in {"no_preconditioner", "sgd"} else preconditioner * grad
                param.add_(direction, alpha=-lr)
                param.add_(gamma * delta, alpha=-1.0 / self.local_steps)
        return loss

    def persistent_state(self):
        v, preconditioner = {}, {}
        for group in self.param_groups:
            for param in group["params"]:
                name = self.names[id(param)]
                v[name] = self.state[param]["v"].detach().cpu().clone()
                preconditioner[name] = self.state[param]["P"].detach().cpu().clone()
        return v, preconditioner
