# Thirty-round diagnostics experiment

Run `bash training_scripts/run_round30_diagnostics.sh` from the code root.
The runner uses GPU 0 and the activated environment's `python` by default;
set `PYTHON_BIN` to override it. See the maintained setup guide.
It runs scaffold_reset, frlora, frlora_scaffold, fedavg, fedsa, scaffold sequentially with identical data,
seed 2025, FP32, eager attention, rank 8 / alpha 16, batch 4, and 5 local epochs.
Learning rate is 2e-4 throughout all 30 rounds. The original cosine round
schedule remains available with `--round_lr_schedule cosine`.

For the published EOS-fixed run, use
[the current reproduction guide](../results/eos_fixed_20260921/README.md).
Older raw output directories mentioned below are not included in Git.

## Reset semantics

The corrected scaffold_reset aggregates trained A/B factors and evaluates
that aggregate. Each subsequent local training starts from the immutable
initial adapter. The SCAFFOLD displacement uses that actual initial adapter,
not the preceding server aggregate. Both control variates persist. Reset is
once per local training call, not once per epoch. The pretrained base stays
fixed. The older output folders used a different, nonaggregating variant;
  do not relabel them as this corrected experiment.

## Measurements

- Epoch: Trainer's training-token loss logs, separate from test loss.
- Round: client training loss, parameter norms, global/initial displacement,
  correction norms, and FedTorch cosine/consistency/drift-diversity formulas.
  These geometry metrics refer to LoRA factor coordinates. FRLoRA additionally
  changes base weights, which these factor-only norms do not measure. A zero
  global factor displacement has undefined drift diversity and is saved null.
- Rounds 5, 10, 15, 20, 25, 30: each post-local model on its own task test;
  the post-aggregation global model on the concatenation of all test records.
  FedSA's whole-test model is explicitly a proxy: shared A plus sample-weighted
  mean of the private B factors. Private states are preserved; this proxy is
  not FedSA's native personalized evaluation. JSON labels this distinction.
- Completion-token mean NLL, exp(NLL) PPL, teacher-forced token accuracy,
  greedy generation (500 new tokens), case-insensitive tokenized exact match,
  ROUGE-L, and the repository's smoothed BLEU-4 implementation.
- LoRA A/B Hessian of completion-token mean NLL, all heldout examples, eval
  at the final round only (`--hessian_every 0`; positive intervals remain available),
  mode and no dropout: SLQ with two Rademacher starts and 20 Lanczos steps,
  full reorthogonalization; four Hutchinson probes for trace. Maximum/minimum
  and top1/top2 are algebraically ordered Ritz estimates, not exact eigenvalues.
  Trace samples and standard error and the SLQ nodes/weights are retained.
  Seeds 2025/2026 and full test support remain fixed between rounds. Spectrum
  plots use Gaussian smoothing with fixed bandwidth across rounds per panel.
  No predictor coupling, subspace leakage, or SGD stability indicators.

Diagnostic RNG state and train/eval mode are restored; diagnostics do not
take optimizer steps. Loss/Hessian score the valid completion tokens within
the original 512-token SFT truncation. Generations allow 500 new tokens after
the prompt. These are distinct length limits.

The source Dolly subset has 80 records; after disjoint 20% holdout and
completion filtering, actual train counts are 15/15/16/16 and test counts
4/3/4/3. Test loss/PPL use token weights; local macro curves average client
metric values and are not pooled global metrics. No independent external
test source is used in this run.

## Outputs

`outputs/llama1b_gpu0/round30_constant_final_hessian/<algorithm>/<run>/`:
args.json, client_partition.json, training_loss.npy, diagnostics.json and
diagnostics.csv, final metrics.json, and figures/*.png / *.pdf.
Predictions/references, spectrum nodes/weights, and trace probes are in JSON.
CSV contains scalars. Full periodic global values are in diagnostics.json;
metrics.json retains the existing final per-task evaluation interface.

Generate figures without training:
`python tools/plot_federated_diagnostics.py outputs/llama1b_gpu0/round30_constant_final_hessian`

The root comparison_figures contains global and local-macro comparisons.
The smoke folder is only a pipeline test (one optimizer step, two generated
tokens, three Lanczos steps) and must not be used for performance claims.
