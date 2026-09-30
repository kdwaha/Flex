# Flex federated instruction tuning

The maintained experiment code lives in [FLEx-8F12](FLEx-8F12/README.md).
It supports task-partitioned Dolly/FLAN instruction tuning with FedAvg,
FedSA-LoRA, FRLoRA, SCAFFOLD variants, and a LoRA adaptation of pFLAlign.

- [Setup and usage](FLEx-8F12/doc/federated_instruction_tuning.md)
- [Completed TinyLlama experiment and reproduction](FLEx-8F12/results/eos_fixed_20260921/README.md)
- [Global and client-specific comparisons](FLEx-8F12/results/eos_fixed_20260921/comparison.md)
- [Hessian estimates and spectrum](FLEx-8F12/results/eos_fixed_20260921/hessian_comparison.md)

The published experiment is exploratory: one seed, four task clients, and
14 held-out examples. It includes an untrained baseline and all 14 trained
conditions, not a selection of favorable results. Task correctness is not
established by token accuracy, BLEU, or ROUGE alone.

Only code, configurations, compact measurements, and selected reports are
versioned. Raw datasets, training logs, model weights, and checkpoints are
excluded. The new experiment scripts disable intermediate and final
checkpoint saving; historical entrypoints retain their original defaults.
