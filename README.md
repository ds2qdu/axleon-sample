# axleon-sample

Scripts for AXLEON MLOps training assets. One folder per asset under `trainjob/`.
Asset runtime: command `bash`, args `-c` plus the one-line clone-and-run below.

## trainjob/pytorch-example

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-example/run.sh   # asset args
MIN_SECONDS=120 EPOCHS=5 STEPS_PER_EPOCH=100 BATCH_SIZE=4096 HIDDEN=4096 LAYERS=4 NPROC_PER_NODE=gpu   # env (defaults)
KUBE_NODE_SIZE KUBE_TRAINJOB_NAME JOB_COMPLETION_INDEX   # set by the platform; run.sh turns them into --nnodes/--node_rank/--master_addr
NPROC_PER_NODE=1 HIDDEN=256 BATCH_SIZE=256 MIN_SECONDS=0 bash trainjob/pytorch-example/run.sh   # quick run by hand (CPU ok)
```

## trainjob/pytorch-glue

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-glue/run.sh   # asset args
S3_SRC=s3://BUCKET/PREFIX   # env, required: prefix holding env.sh, run_glue.py, axleon_progress/
```
