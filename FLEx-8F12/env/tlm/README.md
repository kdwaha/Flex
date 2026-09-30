# tlm environment snapshot

Captured 2026-09-30T18:02:36.131895+00:00 (2026-10-01 in Korea).
This is a separate backup of the current `tlm` package environment.
It does not replace the maintained [flex-frlora setup](../../environment.yml),
and is not a verified lockfile from the original 2026-09-21 experiment date.

## Included files

- `environment.yml`: Conda packages/builds and pip dependencies, without a server-specific prefix.
- `conda-linux-64-explicit.txt`: exact Conda package URLs and MD5 hashes for Linux x86-64.
- `pip-only-requirements.txt`: all 180 pip-managed packages, for installation after the explicit Conda snapshot.
- `pip-freeze-all.txt`: full Python-package inventory, including pip/setuptools/wheel; do not install it on top of the explicit snapshot.
- `runtime.json`: Python/platform versions, Conda/Python inventories, CUDA build, and snapshot limitations.

The recorded runtime is Python 3.10.20, PyTorch 2.5.1+cu124 (CUDA 12.4),
Transformers 4.46.1, PEFT 0.12.0, and TRL 0.9.6.
The Conda export records the torch distribution as 2.5.1; the runtime build
suffix is recorded separately.

## Restore on a compatible Linux x86-64 host

From this directory:

```bash
conda create -n tlm-restored --file conda-linux-64-explicit.txt
conda activate tlm-restored
python -m pip install --no-deps -r pip-only-requirements.txt
export USE_TF=0
export USE_FLAX=0
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

All transitive pip packages are included, so `--no-deps` avoids silently
resolving newer dependency versions. Inspect `python -m pip check` after
restoration; this snapshot preserves an existing environment, not a newly
resolved or conflict-free dependency set. Confirm the printed torch build
matches `runtime.json`; a different CUDA/CPU build is not an identical restore.
The alternative `conda env create -n tlm-restored -f environment.yml` invokes
dependency resolution and may not reproduce identical artifacts.

## Important limitations

- This is package metadata, not an archive of the environment directory.
  Downloads, availability of the recorded package/VCS sources, and a compatible
  NVIDIA driver are still required. The driver is not a Conda package backup.
- BLEURT and editable LLaMA Factory are pinned to the repository commits
  reported by `pip freeze`. Local source edits, untracked files, and editable
  working directories are not backed up here.
- `USE_TF=0` is required for the documented PyTorch workflow to avoid the
  optional TensorFlow/Keras import conflict in this environment.
- Datasets, Hugging Face/model caches, weights, checkpoints, credentials,
  and environment-variable secrets are excluded.
- This is the current snapshot, not proof that all packages matched the
  environment on the original experiment date. Fresh restoration has not
  been run. For Flex experiments, continue to use
  `--save_strategy no --no_save_final_model`.
