# EOS 수정 후 TinyLlama 연합학습 비교

이 폴더는 EOS를 completion loss에 포함하도록 수정한 뒤 완료한 15조건의
공개 스냅샷이다. 학습 전 baseline, 7개 기본 학습 방법, pFLAlign의 7개
ablation을 모두 포함한다. 단일 seed와 작은 test를 사용한 탐색 실험이며,
어떤 방법의 일반적 우월성이나 통계적 유의성을 입증하는 결과가 아니다.

## 결과 보기

- [글로벌과 클라이언트별 비교표 및 곡선](comparison.md)
- [Hessian 비교표 및 그림](hessian_comparison.md)
- [최종 글로벌 및 로컬 평균 CSV](round30_comparison.csv)
- [최종 클라이언트별 CSV](client_comparison.csv)
- [EOS 및 반복 진단 CSV](generation_comparison.csv)
- [Hessian CSV](hessian_comparison.csv)

PNG에는 색뿐 아니라 선 종류, 마커 또는 막대 무늬를 사용한다.
`measurements/`에는 각 조건의 설정, split 크기, 라운드·에폭별 측정값,
Hessian의 Lanczos nodes/weights와 trace samples를 보존했다.
입출력 원문, 중간 학습 로그, 가중치, optimizer 상태는 포함하지 않는다.
`artifact_manifest.json`에는 원본과 공개 파일의 SHA-256을 기록했다.

## 실험 조건

| 항목 | 설정 |
|---|---|
| 모델 | TinyLlama/TinyLlama-1.1B-Chat-v1.0, FP32 |
| 데이터 | Dolly 4 tasks, 원본 추출 80개 |
| 실제 학습과 test | train 62개, test 14개, seed 2025 |
| 클라이언트 | C0 closed_qa, C1 information_extraction, C2 classification, C3 summarization |
| 클라이언트별 train | 15 / 15 / 16 / 16 |
| 클라이언트별 test | 4 / 3 / 4 / 3 |
| 연합학습 | 30 rounds, 매 round 4/4 참여, local epochs 5 |
| 최적화 | learning rate 2e-4 constant, batch 4, accumulation 1 |
| LoRA | r=8, alpha=16, dropout=0.05, q/k/v/o projections |
| 프롬프트와 길이 | Alpaca template, 학습 및 loss 최대 512 tokens |
| 생성 | greedy, 최대 500 new tokens |
| 평가 | 매 5라운드 local 학습 직후 자기 task test, 집계 직후 전체 test |
| Hessian | 마지막 30라운드만, Lanczos 20 steps × 2 probes, trace 4 probes |

SCAFFOLD-reset은 매 local round 시작에 최초 LoRA로 돌아가되 control variates는
유지한다. 학습한 local 모델을 집계하고 평가한 다음 다음 round에서 reset한다.
FedSA의 global 평가는 shared A와 sample-weighted mean B로 구성한 proxy이다.
pFLAlign의 SGD 대조군은 P=1, Delta=0인 SGD이며 AdamW FedAvg와 optimizer가 다르다.
`gamma=1`은 초기 Delta를 매 step Delta/T만큼 빼는 별도 ablation이다.

## 해석 제한

- Test는 동일 소스에서 분리한 held-out records이며 외부 benchmark가 아니다.
  기존 test 결과를 보고 추가 변형을 제안한 탐색 과정이 포함된다. 확증 실험에는
  별도의 validation/test와 여러 seed, 충분한 표본이 필요하다.
- Test loss와 PPL은 실제 response EOS를 포함하지만 prompt와 padding은 제외한다.
  512 tokens에서 잘린 response에 EOS를 인위적으로 추가하지 않는다. EOS를
  제외했던 이전 실험의 loss/PPL과 직접 같은 지표로 비교하면 안 된다.
- Token accuracy는 정답 prefix를 제공하는 teacher-forced 다음 토큰 정답률이지
  instruction 전체의 정답률이 아니다. ROUGE-L/BLEU도 개체와 라벨의 관계를
  잘못 답한 문장에 높은 점수를 줄 수 있다. 이 지표만으로 의미적 정답을 판정하지 않는다.
- Loss는 512-token truncation 안에서, 생성 지표는 최대 500 new tokens와 원래
  reference로 계산한다. 긴 reference의 잘림과 BLEU 길이 패널티를 함께 고려해야 한다.
- Local macro는 클라이언트별 지표의 산술평균이며 PPL도 산술평균이다.
  Global loss는 전체 test의 target-token 가중 평균이고 PPL은 그 값의 exp이다.
  Global ROUGE-L은 예제 평균, BLEU는 합친 corpus 기준이다.
- Hessian은 전체 backbone이 아닌 학습 대상 LoRA A/B의 completion-loss 추정이다.
  Ritz 고윳값과 SLQ density에는 추정 오차가 있다. 비율의 분모가 0에 가까우면
  값이 불안정하며, 곡률이 작다는 사실만으로 좋은 성능을 뜻하지 않는다.
- 연합학습 norm/drift는 LoRA factor 공간 기준이다. FRLoRA의 backbone residual
  변화는 포함하지 않는다. correction_norm은 SCAFFOLD control-variate 통계이지
  pFLAlign의 erf correction 크기가 아니다.

## 재현

저장소 루트에서 `cd FLEx-8F12` 후
[환경 안내](../../doc/federated_instruction_tuning.md)를 따른다.
원본 데이터는 Git에 포함하지 않는다. 소스의 첫 matching records를 task당 20개
추출하는 명령은 다음과 같다. 무작위 추출이 아니며 train/test 분리는 seed 2025이다.

```bash
python tools/download_instruction_subset.py --dataset dolly \
  --tasks closed_qa information_extraction classification summarization \
  --samples-per-task 20 --output data/dolly_tasks_4x20.jsonl
sha256sum data/dolly_tasks_4x20.jsonl
```

사용한 JSONL의 SHA-256은
`28d3b77a94ba30eccf2958fc222c5762837ec7ca43f106d257c772862bfd5a95`이다.
소스는 `databricks/databricks-dolly-15k`이며 모델·데이터의 원래 Hub revision은
별도로 기록하지 않았다. hash가 다르면 같은 데이터라고 가정하지 말고 확인해야 한다.

새 학습을 원할 때만 다음 명령을 실행한다. GPU 0과 모델의 로컬 Hugging Face
cache가 필요하다. runner는 offline mode이므로 처음 사용할 때에는 미리
`huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0`으로 모델을 준비한다.

```bash
OUTPUT_ROOT=outputs/eos_fixed_reproduction \
  bash training_scripts/run_eos_fixed_ablation.sh
OUTPUT_ROOT=outputs/eos_fixed_reproduction PFLALIGN_VARIANT=constant_gamma_one \
  bash training_scripts/run_pflalign_round30.sh
```

기존 실험을 덮어쓰지 않도록 runner에 검사가 있다. 새 실행은 다른 OUTPUT_ROOT를
지정한다. 모든 새 runner는 `--save_strategy no --no_save_final_model`을 사용한다.
공개된 args.json은 당시 저장된 원본 설정이며, 당시에도 diagnostics 경로가
Trainer checkpoint 저장을 비활성화했다.

학습 없이 공개된 측정값에서 표와 그림을 다시 생성할 수도 있다.
생성물은 무시되는 `outputs/` 아래에 두어 공개 스냅샷을 변경하지 않는다.

```bash
mkdir -p outputs/published_report_rebuild
cp -R results/eos_fixed_20260921/measurements outputs/published_report_rebuild/
python tools/build_comparison_report.py outputs/published_report_rebuild
```

표를 재구성할 수 있지만 원문 predictions/references는 제외했으므로 이 배포본만으로
새로운 의미적 정답 지표를 재채점할 수는 없다. 수치와 그림은 원래 실험에서 보존한
자료이며, 공개 준비 과정에서 모델을 재학습하거나 재평가하지 않았다.
