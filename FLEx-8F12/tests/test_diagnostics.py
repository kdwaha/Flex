import unittest
from types import SimpleNamespace

import torch

from evaluation.federated_diagnostics import lanczos, hessian_metrics, token_metrics, diagnostic_state


class QuadraticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.1]))

    def forward(self, **kwargs):
        return SimpleNamespace(loss=(self.x.square()*torch.tensor([-2., 1., 4.])).sum()/2)


class DiagnosticsTests(unittest.TestCase):
    def test_hessian_indefinite_spectrum_and_trace(self):
        model = QuadraticModel()
        batch = dict(labels=torch.tensor([[-100, 1]]))
        before = model.x.detach().clone()
        result = hessian_metrics(model, [batch], steps=3, probes=2, trace_probes=3)
        self.assertAlmostEqual(result['hessian_lambda_max'], 4., places=4)
        self.assertAlmostEqual(result['hessian_lambda_min'], -2., places=4)
        self.assertAlmostEqual(result['hessian_top1_top2'], 4., places=4)
        self.assertAlmostEqual(result['hessian_trace'], 3., places=4)
        for s in result['hessian_spectrum']:
            self.assertAlmostEqual(sum(s['weights']), 1.)
        self.assertTrue(torch.equal(before, model.x))
        self.assertIsNone(model.x.grad)

    def test_token_weighting_and_rng_preservation(self):
        class Uniform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
            def forward(self, labels, **kw):
                return SimpleNamespace(logits=torch.zeros(*labels.shape, 4))
        model = Uniform().train()
        rng = torch.random.get_rng_state()
        with diagnostic_state(model):
            torch.rand(3)
            result = token_metrics(model, [dict(labels=torch.tensor([[-100, 0, 1]])),
                                           dict(labels=torch.tensor([[-100, 2]]))])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertEqual(result['target_tokens'], 3)
        self.assertAlmostEqual(result['perplexity'], 4., places=5)
        self.assertAlmostEqual(result['token_accuracy'], 1/3)


if __name__ == '__main__':
    unittest.main()
