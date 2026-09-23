#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec torchrun train.py
