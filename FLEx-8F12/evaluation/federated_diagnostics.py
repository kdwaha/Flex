"""Read-only periodic FL diagnostics; completion-token metrics and matrix-free Hessian."""
import csv
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.instruction_metrics import evaluate_instruction_generations, metric_tokenize


@contextmanager
def diagnostic_state(model):
    training = model.training
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            yield
        finally:
            model.train(training)
            random.setstate(py_state)
            np.random.set_state(np_state)


def completion_batches(tokenizer, formatter, collator, dataset, max_length):
    batches = []
    # One example at a time: no pad/eos ambiguity; token weights are exact.
    for row in dataset:
        text = formatter({key: [value] for key, value in row.items()})[0]
        encoded = tokenizer(text, truncation=True, max_length=max_length)
        batch = collator([encoded])
        if (batch['labels'][:, 1:] != -100).sum().item() == 0:
            raise ValueError('Test example has no scored completion tokens')
        batches.append(batch)
    return batches


def token_metrics(model, batches):
    nll, correct, count = 0., 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in batches:
            inputs = {key: value.to(device) for key, value in batch.items()}
            out = model(**inputs, use_cache=False)
            labels = inputs['labels'][:, 1:]
            logits = out.logits[:, :-1].float()
            valid = labels != -100
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=-100, reduction='sum').item()
            correct += ((logits.argmax(-1) == labels) & valid).sum().item()
            count += valid.sum().item()
    loss = nll / count
    return dict(test_loss=loss, perplexity=math.exp(loss), token_accuracy=correct/count,
                nll_sum=nll, target_tokens=count, correct_tokens=correct)


def lanczos(operator, size, device, steps, probes, seed=2025):
    """SLQ nodes and quadrature weights; full reorthogonalization of Krylov basis."""
    generator = torch.Generator(device=device).manual_seed(seed)
    spectra = []
    for _ in range(probes):
        q = torch.randint(0, 2, (size,), device=device, generator=generator).float().mul_(2).sub_(1)
        q /= q.norm()
        basis, alphas, betas = [], [], []
        for j in range(min(steps, size)):
            basis.append(q)
            z = operator(q)
            if j:
                z = z - betas[-1] * basis[-2]
            alpha = torch.dot(q, z)
            z = z - alpha * q
            # Two passes avoid ghosts / duplicated leading Ritz values.
            for _pass in range(2):
                for u in basis:
                    z = z - torch.dot(u, z) * u
            beta = z.norm()
            alphas.append(float(alpha))
            if beta.item() < 1e-7 or j == min(steps, size)-1:
                break
            betas.append(float(beta))
            q = z / beta
        tri = torch.diag(torch.tensor(alphas, dtype=torch.float64))
        if len(alphas) > 1:
            off = torch.tensor(betas[:len(alphas)-1], dtype=torch.float64)
            tri += torch.diag(off, 1) + torch.diag(off, -1)
        eig, vec = torch.linalg.eigh(tri)
        spectra.append(dict(nodes=eig.tolist(), weights=vec[0].square().tolist()))
    return spectra


def hessian_metrics(model, batches, steps=20, probes=2, trace_probes=4):
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    sizes = [p.numel() for p in params]
    size = sum(sizes)
    total_tokens = sum(int((b['labels'][:, 1:] != -100).sum()) for b in batches)

    def operator(vector):
        pieces = [x.view_as(p) for x, p in zip(vector.split(sizes), params)]
        result = torch.zeros_like(vector)
        # Recompute graphs per HVP/sample to bound memory on the 1B model.
        with torch.enable_grad():
            for batch in batches:
                inputs = {k: v.to(device) for k, v in batch.items()}
                tokens = (inputs['labels'][:, 1:] != -100).sum().item()
                loss = model(**inputs, use_cache=False).loss * (tokens/total_tokens)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                dot = sum((g*v).sum() for g, v in zip(grads, pieces))
                hv = torch.autograd.grad(dot, params)
                result.add_(torch.cat([v.detach().reshape(-1) for v in hv]))
                del loss, grads, dot, hv
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite Hessian-vector product')
        return result

    spectra = lanczos(operator, size, device, steps, probes)
    # Select the probe that resolved the largest algebraic Ritz eigenvalue.
    best = max(spectra, key=lambda s: s['nodes'][-1])['nodes']
    top1, top2 = best[-1], best[-2] if len(best) > 1 else None
    generator = torch.Generator(device=device).manual_seed(2026)
    traces = []
    for _ in range(trace_probes):
        z = torch.randint(0, 2, (size,), device=device, generator=generator).float().mul_(2).sub_(1)
        traces.append(float(torch.dot(z, operator(z))))
    return dict(hessian_lambda_max=top1, hessian_lambda_min=min(s['nodes'][0] for s in spectra),
                hessian_lambda_2=top2,
                hessian_top1_top2=top1/top2 if top2 is not None and abs(top2)>1e-10 else None,
                hessian_trace=float(np.mean(traces)), hessian_trace_samples=traces,
                hessian_trace_se=float(np.std(traces, ddof=1)/math.sqrt(len(traces))) if len(traces)>1 else None,
                hessian_spectrum=spectra, hessian_parameter_count=size,
                hessian_examples=len(batches), hessian_target_tokens=total_tokens,
                lanczos_steps=steps, lanczos_probes=probes, trace_probes=trace_probes,
                hessian_spec='LoRA A/B completion-token mean NLL; SLQ Ritz estimates; Hutchinson trace; all test examples')


def flat(state):
    return torch.cat([v.detach().float().reshape(-1) for v in state.values()])


class FederatedDiagnostics:
    def __init__(self, output_dir, args, tokenizer, formatter, collator, datasets, fed_alg, num_rounds):
        self.path, self.args = Path(output_dir), args
        self.fed_alg = fed_alg
        self.num_rounds = num_rounds
        self.tokenizer, self.datasets = tokenizer, datasets
        self.batches = [completion_batches(tokenizer, formatter, collator, ds, args.max_length) for ds in datasets]
        self.records = []

    def due(self, round_id):
        return round_id % self.args.diagnostics_every == 0 or round_id == self.num_rounds

    def evaluate(self, model, round_id, client, measure_hessian=True):
        dataset = list(self.datasets[client]) if client >= 0 else [row for ds in self.datasets for row in ds]
        batches = self.batches[client] if client >= 0 else [b for bs in self.batches for b in bs]
        with diagnostic_state(model):
            metrics = token_metrics(model, batches)
            generated = evaluate_instruction_generations(
                model, self.tokenizer, dataset, template_name=self.args.template,
                batch_size=self.args.generation_eval_batch_size,
                max_new_tokens=self.args.generation_max_new_tokens,
                max_prompt_length=self.args.max_length, include_predictions=True)
            generated['exact_match'] = sum(metric_tokenize(a)==metric_tokenize(b) for a,b in
                                           zip(generated['predictions'], generated['references'])) / len(dataset)
            metrics.update(generated)
            if measure_hessian and (round_id == self.num_rounds or (self.args.hessian_every > 0 and round_id % self.args.hessian_every == 0)):
                metrics.update(hessian_metrics(model, batches, self.args.hessian_lanczos_steps,
                                               self.args.hessian_probes, self.args.hessian_trace_probes))
        self.records.append(dict(kind='evaluation', round=round_id, client=client,
                                 task=dataset[0].get('category') if client>=0 else 'all_test',
                                 model_definition=('shared_A_sample_weighted_B_global_proxy'
                                     if client < 0 and self.fed_alg == 'fedsa' else
                                     'trained_local' if client >= 0 else 'aggregated_global'),
                                 phase='post_local' if client>=0 else 'post_aggregation', **metrics))
        self.save()
        print(f'DIAGNOSTICS round={round_id} client={client} loss={metrics["test_loss"]:.4f} '
              f'ppl={metrics["perplexity"]:.4f}', flush=True)

    def local(self, model, client, round_id, trainer, current, previous, start, auxiliary, lr):
        v, g, s = flat(current), flat(previous), flat(start)
        self.records.append(dict(kind='local_update', round=round_id, client=client,
            train_loss=trainer.state.log_history[-1].get('train_loss'), learning_rate=lr,
            weight_norm=float(v.norm()), cosine_similarity=float(F.cosine_similarity(v, g, dim=0)),
            local_update_norm=float((v-s).norm()), global_distance=float((v-g).norm()),
            correction_norm=float(flat(auxiliary).norm()) if auxiliary else 0.))
        for item in trainer.state.log_history:
            if 'loss' in item:
                self.records.append(dict(kind='epoch', round=round_id, client=client,
                                         local_epoch=item.get('epoch'), train_loss=item['loss']))
        self.save()
        if self.due(round_id):
            self.evaluate(model, round_id, client)

    def server(self, model, round_id, previous, current, locals_, selected, auxiliary):
        g, old = flat(current), flat(previous)
        consistency = sum(float((flat(locals_[i])-old).square().sum()) for i in selected)/len(selected)
        denominator = float((g-old).square().sum())
        self.records.append(dict(kind='server_update', round=round_id, client=-1,
            weight_norm=float(g.norm()), consistency=consistency,
            drift_diversity=consistency/denominator if denominator>1e-20 else None,
            global_update_norm=math.sqrt(denominator),
            correction_norm=float(flat(auxiliary).norm()) if auxiliary else 0.))
        self.save()
        if self.due(round_id):
            self.evaluate(model, round_id, -1)

    def save(self):
        with (self.path/'diagnostics.json').open('w') as f:
            json.dump(self.records, f, indent=2, allow_nan=False)
        scalar = [{k:v for k,v in r.items() if not isinstance(v, (list,dict))} for r in self.records]
        keys = sorted({k for row in scalar for k in row})
        with (self.path/'diagnostics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(scalar)
