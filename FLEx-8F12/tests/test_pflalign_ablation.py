import unittest
import torch
from federated_learning.pflalign import PFLAlignOptimizer


class AblationTests(unittest.TestCase):
    def test_variants_match_explicit_first_step(self):
        for variant in ('full', 'no_preconditioner', 'no_correction', 'constant_gamma',
                        'constant_gamma_one', 'hard_gamma', 'no_personalization', 'sgd'):
            with self.subTest(variant=variant):
                p = torch.nn.Parameter(torch.tensor([1., 1., 1.]))
                delta = torch.tensor([.4, -.4, 0.])
                opt = PFLAlignOptimizer([('w', p)], .2, .9, 1e-12, 5,
                    {'w': delta}, {'w': torch.zeros(3)}, {'w': torch.zeros(3)}, variant)
                p.grad = torch.full((3,), 2.)
                opt.step()
                gamma = .5 + .5 * torch.erf(torch.tensor(.2 / (2*.36)**.5)) * delta.sign()
                if variant in ('no_correction', 'no_personalization', 'sgd'):
                    gamma = torch.zeros(3)
                elif variant == 'constant_gamma':
                    gamma = torch.full((3,), .5)
                elif variant == 'constant_gamma_one':
                    gamma = torch.ones(3)
                elif variant == 'hard_gamma':
                    gamma = .5 + .5 * delta.sign()
                factor = 1. if variant in ('no_preconditioner', 'sgd') else .01
                torch.testing.assert_close(p, 1 - .2*factor*2 - gamma*delta/5)

    def test_gamma_one_cancels_initial_delta_over_actual_local_steps(self):
        # Current data: ceil(15 or 16 samples / batch 4) * 5 epochs = 20 steps.
        for zero_gradient in (True, False):
            base = torch.tensor([1., -2., .5], dtype=torch.float64)
            delta = torch.tensor([.4, -.8, 0.], dtype=torch.float64)
            p = torch.nn.Parameter(base + delta)
            opt = PFLAlignOptimizer([('w', p)], .2, .9, 1e-12, 20,
                {'w': delta}, {'w': torch.zeros_like(p)}, {'w': torch.zeros_like(p)},
                'constant_gamma_one')
            gradient_update = torch.zeros_like(p)
            for step in range(20):
                p.grad = torch.zeros_like(p) if zero_gradient else torch.tensor(
                    [step + 1., -2., .5], dtype=p.dtype)
                opt.step()
                gradient_update.add_(opt.state[p]['P'] * p.grad, alpha=-.2)
            torch.testing.assert_close(p, base + gradient_update, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(delta, torch.tensor([.4, -.8, 0.], dtype=p.dtype))

    def test_sgd_matches_two_steps_without_momentum(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = PFLAlignOptimizer([('w', p)], .2, .9, 1e-12, 5,
            {'w': torch.zeros(1)}, {'w': torch.zeros(1)}, {'w': torch.zeros(1)}, 'sgd')
        for grad in (2., -1.):
            p.grad = torch.tensor([grad])
            opt.step()
        torch.testing.assert_close(p, torch.tensor([.8]))
