#!/usr/bin/env bash
set -e

pip install -r requirements.txt

python prepare_ocrepair_sft.py
python train_ocrepair_lora.py
