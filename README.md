# Post-OCR Correction Fine-tuning

This repository contains scripts for fine-tuning an instruction-tuned language model for post-OCR correction on the English Overproof subset of the HIPE-OCRepair 2026 benchmark.

The current setup fine-tunes `Qwen/Qwen2.5-3B-Instruct` with LoRA on noisy OCR / ground-truth correction pairs.

## Dataset

We use the English Overproof data from:

```text
https://github.com/hipe-eval/HIPE-OCRepair-2026-data/tree/main/data/v0.9/overproof/en

hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_train_en.jsonl
hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_dev_en.jsonl
hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_test_en.jsonl
```

```json
{
  "document_metadata": {...},
  "ground_truth": {
    "transcription_unit": "corrected text"
  },
  "ocr_hypothesis": {
    "transcription_unit": "noisy OCR text"
  },
  "ocr_postcorrection_output": {
    "transcription_unit": "None"
  }
}
```

For training, we use:

* ocr_hypothesis.transcription_unit as input
* ground_truth.transcription_unit as target

The current split sizes are:
```text
train: 146 examples
validation: 30 examples
test: 32 examples
``` 

### Task

The model is trained to correct OCR errors in historical English newspaper text.

The instruction format is:
```text
System:
You are an OCR post-correction system for historical English newspapers. Correct OCR errors while preserving the original wording, punctuation, line breaks, spelling style, names, dates, and historical language. Do not modernize the text. Return only the corrected text.

User:
Correct the following OCR text from a historical newspaper.

OCR text:
<noisy OCR text>

Assistant:
<ground-truth corrected text>
```

### Baseline

Before fine-tuning, the script computes the OCR baseline on the validation set by comparing the original OCR hypothesis against the ground truth.

Current validation baseline:
```text
OCR CER: 0.087017
OCR WER: 0.344423
```

## Results

We report Character Error Rate (CER) and Word Error Rate (WER). Lower is better.

| System | Model | Training | Split | CER ↓ | WER ↓ | Notes |
|---|---|---|---|---:|---:|---|
| OCR baseline | Original OCR hypothesis | None | Validation | 0.087017 | 0.344423 | No correction applied |
| LoRA fine-tuned | Qwen/Qwen2.5-3B-Instruct | LoRA, 3 epochs | Validation | TBD | TBD | Generated post-correction |
| LoRA fine-tuned | Qwen/Qwen2.5-3B-Instruct | LoRA, 3 epochs | Test | TBD | TBD | Final test evaluation |

