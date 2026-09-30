# Federated instruction tuning

This revision adds a task-per-client SFT path for Dolly/FLAN, FedSA-LoRA, and
the ICLR 2025 FRLoRA update rule.  It also saves a deterministic task partition
and final held-out completion loss in each run directory. It additionally
supports SCAFFOLD/reset and pFLAlign ablations. The current
[EOS-fixed experiment](../results/eos_fixed_20260921/README.md) is the public
result reference; the CPU checks below are historical pipeline checks.

## Environment

The publication checks use Python 3.10, PyTorch 2.5.1+cu124,
Transformers 4.46.1, PEFT 0.12.0, TRL 0.9.6, and Matplotlib 3.10.9.
The earlier CPU smoke checks used an older PyTorch/Transformers/PEFT stack.
For the maintained environment, from `FLEx-8F12`:

```bash
conda env create -f environment.yml
conda activate flex-frlora
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Choose a PyTorch wheel compatible with the host driver; on CPU use the CPU
wheel index instead of `cu124`. Dependency installation may initially resolve
a different PyTorch build; the explicit install above pins the tested version.
FRLoRA is deliberately full-precision; do not
pass `--load_in_4bit` or `--load_in_8bit` to it.

The six new experiment scripts disable checkpoints with
`--save_strategy no --no_save_final_model`. Direct calls to `main_sft.py`
must pass both options to obtain the same behavior; legacy defaults remain
unchanged. `save_model_freq` is not a replacement for the Trainer save strategy.
Diagnostics, scalar measurements, and plots are still saved.

Run CPU-only unit tests without downloading model weights:

```bash
USE_TF=0 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -q
```

## Task data

Download four equal-sized Dolly tasks (the output remains local; it is never
uploaded by the federated loop):

```bash
python tools/download_instruction_subset.py --dataset dolly \
  --tasks closed_qa information_extraction classification summarization \
  --samples-per-task 600 --output data/dolly_tasks_4x600.jsonl
```

`--dataset_name dolly --local_data_dir ...` accepts the original Dolly schema
(`instruction`, `context`, `response`, `category`) and common local variants
(`instruction`, `input`, `output`, `type/category`).  FLAN data is normalized
from `inputs`, `targets`, and `task_name`/`task` in the same way.  To list
source-task names without downloading the 3.5M-example aggregate FLAN split:

```bash
python tools/download_instruction_subset.py --dataset flan --list-tasks
```

Then request four names (for example `cb_10templates`, `copa_10templates`,
`rte_10templates`, `cola_10templates`) with `--dataset flan --tasks ...
--output ...`.  The downloader fetches only those individual task files and
adds their source task as `task_name`.

## Comparable experiment

Use the exact same dataset, client assignment, seed, rounds, rank, and local
steps for all three methods:

```bash
bash training_scripts/run_dolly_task_baselines.sh
python tools/summarize_metrics.py outputs/dolly_task_baselines
```

The shell script runs FedAvg, FedSA, and FRLoRA on four task clients.  Its final
metric is the sample-weighted held-out completion loss.  FedSA additionally
reports each personalized client `(shared A, private B_i)` loss. Private states
remain in memory when checkpoint saving is disabled. The diagnostic global
comparison uses shared A plus sample-weighted mean B, explicitly labeled as
a proxy rather than FedSA's native personalized model.
`training_scripts/run_flan_task_baselines.sh` provides the same comparison for
the four FLAN source tasks shown above.

## Client-task test and personalized generation metrics

By default, `--eval_fraction` creates a deterministic **record-disjoint**
held-out test partition inside every client task: a client trains only on its
own task's train records and is evaluated only on its own task's held-out
records.  It is a hold-out split from the same source file, not an independently
sourced benchmark.  For a separately prepared test file/directory, pass
`--test_data_dir`; that data is partitioned by the same task-to-client mapping
and is never supplied to a local trainer.

Add the following flags to a run to calculate deterministic prompt-only
ROUGE-L and BLEU-4 after every client update, before aggregation:

```bash
--generation_eval --post_local_generation_eval \
--generation_max_new_tokens 128 --generation_eval_batch_size 4
```

For a long-generation final personalized evaluation only (for example 500
tokens after the last local update of a multi-round run), add
`--post_local_generation_eval_final_round_only`.

`metrics.json` contains final task-wise generation scores when
`--generation_eval` is enabled.  `post_local_personalized_metrics.json` and
`metrics.json:post_local_personalized_evaluation` contain all pre-aggregation
client/round records plus each client's latest score and macro/example-weighted
averages.  `rougeL_f1` is in `[0, 1]`; `bleu4_pct` is corpus BLEU-4 in
`[0, 100]`.

```bash
python tools/summarize_metrics.py outputs/your_run_root --generation-phase post-local
python tools/summarize_metrics.py outputs/your_run_root --generation-phase final
```

The pre-aggregation state is intentionally method-specific: FedAvg evaluates a
client's just-trained local adapter; FedSA evaluates shared A plus that client's
private B; FRLoRA evaluates the current residual base plus that client's just-
trained factors.  The FRLoRA measurement must occur before aggregation because
aggregation folds the residual into the base and resets LoRA factors.

## FRLoRA fidelity

For each LoRA target weight `W0` and scale `s = alpha/r`, FRLoRA does exactly:

1. Once: top-r SVD creates `B0,A0`, then stores `W_hat0 = W0 - s B0 A0`.
2. Each round: average local `B` and `A` independently, add
   `s * (Bbar Abar - B0 A0)` to the frozen base weights.
3. Reset all LoRA factors to `B0,A0` before the next round.

An adapter-only checkpoint cannot represent the accumulated frozen-base
residuals. If saving is explicitly enabled for another experiment, the final
FRLoRA checkpoint must therefore be a merged full model. The published suite
and the new runners do not save such checkpoints.

## Historical CPU smoke comparison

These execution checks predate the EOS-preserving loss fix. Do not compare
their loss values directly with the published EOS-fixed results.

On this checkout, a functional CPU test used
`hf-internal-testing/tiny-random-LlamaForCausalLM`, 80 real Dolly examples
(four task clients × 20), 10 rounds, two local steps, rank 4, and the same
seed/held-out partition.  Final weighted held-out completion loss was:

| Method | Loss |
| --- | ---: |
| FedAvg-LoRA | 10.377871 |
| FRLoRA | 10.377779 |

FRLoRA was lower by `0.000092` in this **implementation smoke test**.  The
tiny random model and 16 held-out examples are far too small to support a
scientific effectiveness claim; use the GPU script above (and multiple seeds)
to assess whether the improvement persists on a real instruction model.

## Historical TinyLlama 1.1B CPU run

The public Llama-family checkpoint `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was
also run in FP32 on CPU with a deterministic 4-client task split (20 examples
per client, 16 train / 4 held out), 10 rounds, one local step, rank 4, and
seed 2025.  This confirms the real 1B model path for all three algorithms.

| Dataset | FedAvg-LoRA | FedSA-LoRA (personalized) | FRLoRA |
| --- | ---: | ---: | ---: |
| Dolly: closed QA / information extraction / classification / summarization | 0.940042 | 0.887353 | 0.987429 |
| FLAN: CB / COPA / RTE / CoLA source tasks | 1.265760 | — | 1.628270 |

The metric is weighted held-out completion loss (lower is better).  On these
very small, one-seed CPU runs, FRLoRA did **not** improve on FedAvg (Dolly
`+0.047387`, FLAN `+0.362510`).  FedSA's Dolly value uses its intended
personalized `(shared A, private B_i)` evaluation, so it is not a direct global
model comparison.  These are execution checks, not a conclusive method result;
a GPU run with more task data, local steps, rounds, and repeated seeds is still
required for a valid effectiveness claim.
