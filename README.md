# axleon-sample

AXLEON MLOps 의 `training` 환경 자산이 쓰는 학습 샘플. `trainjob/` 아래 자산 이름과 같은 폴더 하나씩이고,
자산의 runtime 은 `bash -c "git clone … && bash /tmp/repo/trainjob/<이름>/run.sh"` 한 줄이다.

## 두 샘플의 차이

| | pytorch-example | pytorch-glue |
|---|---|---|
| 목적 | **플랫폼 점검**. GPU 잡기 · torchrun · 멀티 GPU · 멀티 노드(DDP)가 되는지 | **진짜 학습 예시**. 사전학습 모델을 데이터로 fine-tuning |
| 모델 · 데이터 | 작은 MLP, 데이터는 메모리에서 난수로 생성(`y = sum(x)`) | DistilBERT, GLUE SST-2(영화평 긍정 · 부정, 6.7만 문장)를 HuggingFace Hub 에서 받음 |
| 도는 시간 | `MIN_SECONDS`(120초) 이상 | `EPOCHS`(3), 몇 분 |
| 필요한 네트워크 | GitHub | GitHub · PyPI · HuggingFace Hub |
| 출력 | epoch 마다 loss | epoch 마다 loss · 정확도 · F1 |

공통: torchrun 래퍼가 같아서 워크로드 폼의 Workers 를 2 이상으로 주면 코드 수정 없이 다중 노드로 돈다
(플랫폼 env `KUBE_NODE_SIZE` · `KUBE_TRAINJOB_NAME` · `JOB_COMPLETION_INDEX` 를 읽음).
MLflow 는 `MLFLOW_TRACKING_URI` 가 있고 `mlflow` 가 설치돼 있을 때만 기록한다.

## trainjob/pytorch-example

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-example/run.sh   # 자산 args
MIN_SECONDS=120 EPOCHS=5 STEPS_PER_EPOCH=100 BATCH_SIZE=4096 HIDDEN=4096 LAYERS=4 NPROC_PER_NODE=gpu   # env 기본값
NPROC_PER_NODE=1 HIDDEN=256 BATCH_SIZE=256 MIN_SECONDS=0 bash trainjob/pytorch-example/run.sh   # 손으로 빨리 돌려보기 (CPU 가능)
```

바꿔 쓰는 것(env): `MIN_SECONDS`(얼마나 오래), `BATCH_SIZE` · `HIDDEN` · `LAYERS`(GPU 를 얼마나 세게), `NPROC_PER_NODE`(`gpu` = pod 의 GPU 전부).

## trainjob/pytorch-glue

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-glue/run.sh   # 자산 args
TASK=sst2 MODEL_NAME=distilbert/distilbert-base-uncased EPOCHS=3 BATCH_SIZE=32 MAX_SEQ_LENGTH=128 NPROC_PER_NODE=gpu   # env 기본값
HF_ENDPOINT=https://<mirror>   # env, 선택: pod 에서 HuggingFace Hub 로 못 나갈 때 미러
```

바꿔 쓰는 것(env): `TASK`(`sst2` · `mrpc` · `cola` · `qqp` · `qnli` · `rte`), `MODEL_NAME`(Hub 의 텍스트 모델, `소유자/이름` 형식),
`EPOCHS` · `BATCH_SIZE` · `MAX_SEQ_LENGTH`. 모델이나 과제를 키우면 더 오래, 더 세게 GPU 를 쓴다.

## 내 스크립트로 바꾸기

`trainjob/<자산 이름>/run.sh` 를 추가하고(다중 노드는 기존 `run.sh` 의 torchrun 블록을 복사) 커밋한 뒤,
자산 args 에서 폴더 이름만 바꾼다. args 는 한 줄로 둔다.
