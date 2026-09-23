#!/usr/bin/env bash
set -eu
: "${S3_SRC:?S3_SRC must be the s3:// prefix holding env.sh, run_glue.py and axleon_progress/}"
S3_SRC="${S3_SRC%/}"

python3 -m pip install --no-cache-dir --root-user-action=ignore awscli

python3 -m awscli s3 cp "$S3_SRC/env.sh" /workspace/env.sh
python3 -m awscli s3 cp "$S3_SRC/run_glue.py" /workspace/run_glue.py
python3 -m awscli s3 cp "$S3_SRC/axleon_progress/" /tmp/axleon/ --recursive

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r /tmp/axleon/axleon_progress "$SITE_PACKAGES/axleon_progress"
cp /tmp/axleon/axleon_progress.pth "$SITE_PACKAGES/axleon_progress.pth"

chmod +x /workspace/env.sh
exec /workspace/env.sh
