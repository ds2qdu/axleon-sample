# axleon-sample

Scripts for AXLEON MLOps training assets. One folder per asset under `trainjob/`.
Asset runtime: command `bash`, args `-c` plus the one-line clone-and-run below.

## trainjob/pytorch-example

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-example/run.sh   # asset args
EPOCHS=3 STEPS_PER_EPOCH=50 BATCH_SIZE=64   # env (defaults in train.py)
torchrun trainjob/pytorch-example/train.py   # run by hand
```

## trainjob/pytorch-glue

```bash
git clone --depth=1 https://github.com/ds2qdu/axleon-sample.git /tmp/repo && bash /tmp/repo/trainjob/pytorch-glue/run.sh   # asset args
S3_SRC=s3://BUCKET/PREFIX   # env, required: prefix holding env.sh, run_glue.py, axleon_progress/
```
