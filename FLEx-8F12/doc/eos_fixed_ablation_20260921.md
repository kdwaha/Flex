# EOS 수정 후 재실험

실행: `bash training_scripts/run_eos_fixed_ablation.sh`

공개 결과: [통합 비교](../results/eos_fixed_20260921/comparison.md).
재현 방법과 해석 제한은 [공개 결과 안내](../results/eos_fixed_20260921/README.md)에 정리했다.
실행 시에는 `outputs/llama1b_gpu0/eos_fixed_ablation_20260921/report/` 아래에
표·PNG·PDF·CSV를 생성하며, 기존 결과가 있으면 덮어쓰지 않는다.

## 공통 조건

- TinyLlama/TinyLlama-1.1B-Chat-v1.0, GPU 0, float32.
- Dolly 4 task clients: closed_qa, information_extraction, classification, summarization.
- 기존 split seed 2025: train 62, test 14 (4/3/4/3). Test 데이터는 학습 loss에 사용하지 않는다.
- 30 rounds × 5 local epochs, batch 4, accumulation 1, LR 2e-4 constant.
- LoRA r=8, alpha=16, dropout=.05, q/k/v/o projections, max_length=512.
- 기존 Alpaca prompt 유지; greedy generation 최대 500 new tokens.
- 5라운드마다 local 학습 직후 자기 task test, aggregation 직후 global 전체 test.
- Loss/PPL/token accuracy/Hessian: 실제 response EOS 포함, prompt/padding -100.
  512 길이에서 잘린 response에는 인위적으로 EOS를 추가하지 않는다.
- Hessian은 마지막 30라운드에만, 모든 학습 조건에 Lanczos 20×2, trace 4 probes.
  LoRA Hessian Ritz 최대/최소/두 번째 고유값, trace, 두 비율, SLQ spectrum density.
  학습 전 baseline은 기존처럼 Hessian 제외.
- Loss/ROUGE-L/BLEU/PPL/token accuracy/Exact match, 클라이언트·epoch 곡선, 기존 FL 진단.
- 추가 관측: EOS 종료율, length cap 비율, 평균 생성 길이, 중복 4-gram 비율,
  BLEU BP 및 unsmoothed 1~4-gram precision. 원문 predictions/references도 저장.
- 모든 학습 체크포인트와 최종 모델 저장 비활성화; 일시적인 파라미터 상태는 메모리에만 유지.

## 비교 조건

학습 전 baseline + FedAvg, FedSA, FRLoRA, FRLoRA+SCAFFOLD, SCAFFOLD,
SCAFFOLD-reset, pFLAlign full + 아래 7 ablations (학습 전 baseline 포함 총 15조건).
기본 suite는 14조건을 실행하며 gamma=1은 아래 추가 명령으로 실행한다.

| pFLAlign variant | 변경 | 유지 |
|---|---|---|
| no_preconditioner | update의 P=1 | personalized init, m/v 기반 erf correction |
| no_correction | gamma=0 | personalized init, P |
| constant_gamma | gamma=.5 | personalized init, P |
| constant_gamma_one | gamma=1 | personalized init, P |
| hard_gamma | gamma=(1+sign(m*Delta))/2, tie=.5 | personalized init, P |
| no_personalization | local 시작 Delta=0, correction=0 | P |
| sgd | P=1, Delta=0, correction=0 | 같은 SGD step size, 집계 |

모든 pFLAlign 변형은 clipping 없이 beta=.9, epsilon=1e-12,
m은 매 round 재시작, v/P는 client별 유지한다. 사용하지 않는 P도 재귀식은 유지한다.
SGD 대조군은 momentum 없는 SGD이며 기존 AdamW FedAvg와 optimizer가 다르다.
no_personalization은 init와 correction을 함께 없애는 block ablation이다.
SCAFFOLD-reset은 local round 시작 시 초기 LoRA로 reset하고 controls는 유지한다.
학습된 local을 집계·평가한 뒤 다음 round에서 reset하며 epoch마다 reset하지 않는다.

## 해석 제한

단일 seed, test 14개이므로 ablation 효과는 탐색적 관찰이며 유의한 일반화 개선으로 단정하지 않는다.
기존 긴 reference/512 loss truncation/500 generation cap은 EOS 효과를 분리하기 위해 그대로 둔다.
새 loss/PPL은 이전 EOS 제외 실험과 직접 동일 정의로 비교할 수 없다.
FedSA global은 shared A + sample-weighted B proxy이다.
기존 correction_norm은 SCAFFOLD controls이며 pFLAlign erf correction norm이 아니다.

## 추가 실험: constant gamma=1 (2026-09-22)

`constant_gamma_one` 조건을 추가해 완료했다. 기존 full의 개인화 초기화와
P, m/v/P 상태 관리, 데이터, seed, 학습·평가 조건을 그대로 유지하고 gamma 전체만 1로 고정한다.
각 optimizer step에서 Delta/T를 빼므로 T step의 correction 합은 Delta이다.
현재 각 client는 ceil(15 또는 16 / 4) × 5 = 20 optimizer steps를 사용한다.
초기 Delta의 직접 덧셈은 상쇄되지만 gradient 경로에 미치는 영향은 유지된다.
기존 gamma=.5, hard-sign, full 결과는 변경하거나 재학습하지 않는다.

실행 명령:

```sh
OUTPUT_ROOT=outputs/llama1b_gpu0/eos_fixed_ablation_20260921 \
PFLALIGN_VARIANT=constant_gamma_one bash training_scripts/run_pflalign_round30.sh
```

공개 통합 report에는 `pFLAlign: gamma=1` 행이 포함되어 있다.
실행은 체크포인트를 저장하지 않으며 GPU 0을 사용한다.
