import torch
import evaluate
from jiwer import wer
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "overproof_ocrepair_sft"

OUTPUT_DIR = "qwen2_5_3b_overproof_ocrepair_lora"
HUB_MODEL_ID = "emanuelaboros/qwen2-5-3b-overproof-postcorrection"

NUM_EPOCHS = 10
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
MAX_LENGTH = 4096

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

cer_metric = evaluate.load("cer")


def formatting_func(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def compute_cer_wer(predictions, references):
    predictions = [str(x) if x is not None else "" for x in list(predictions)]
    references = [str(x) if x is not None else "" for x in list(references)]

    cer = cer_metric.compute(predictions=predictions, references=references)
    word_error_rate = wer(references, predictions)

    return cer, word_error_rate


def print_initial_ocr_baseline(eval_dataset):
    ocr_texts = list(eval_dataset["ocr"])
    references = list(eval_dataset["ground_truth"])

    cer, word_error_rate = compute_cer_wer(ocr_texts, references)

    print("=" * 100)
    print("INITIAL OCR BASELINE ON VALIDATION SET")
    print(f"Samples: {len(eval_dataset)}")
    print(f"OCR CER: {cer:.6f}")
    print(f"OCR WER: {word_error_rate:.6f}")
    print("=" * 100, flush=True)

    return {
        "ocr_val_cer": cer,
        "ocr_val_wer": word_error_rate,
    }


def format_metric(value):
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_and_upload_model_card(
    hub_model_id,
    output_dir,
    base_model,
    train_examples,
    validation_examples,
    test_examples,
    ocr_baseline_metrics,
    final_eval_metrics=None,
):
    final_eval_metrics = final_eval_metrics or {}

    baseline_cer = format_metric(ocr_baseline_metrics.get("ocr_val_cer"))
    baseline_wer = format_metric(ocr_baseline_metrics.get("ocr_val_wer"))
    eval_loss = format_metric(final_eval_metrics.get("eval_loss"))
    eval_token_accuracy = format_metric(
        final_eval_metrics.get("eval_mean_token_accuracy")
    )

    readme = f"""---
language:
- en
library_name: peft
base_model: {base_model}
tags:
- ocr
- post-ocr-correction
- historical-documents
- historical-newspapers
- hipe-ocrepair-2026
- overproof
- lora
- peft
pipeline_tag: text-generation
---

# Qwen2.5-3B Overproof Post-OCR Correction

LoRA adapter for `{base_model}`, fine-tuned for post-OCR correction of historical English newspaper text.

The model corrects noisy OCR while trying to preserve historical spelling, wording, punctuation, names, dates, and line breaks.

## Data

English Overproof subset of HIPE-OCRepair 2026.

| Split | Examples |
|---|---:|
| Train | {train_examples} |
| Validation | {validation_examples} |
| Test | {test_examples} |

Training pairs:

```text
Input:  ocr_hypothesis.transcription_unit
Target: ground_truth.transcription_unit
```

The prompt uses available metadata such as date, language, publication title, document type, and segmentation source. It does **not** use CER, WER, OCR quality scores, or any ground-truth-derived information.

## Training

| Parameter | Value |
|---|---:|
| Method | LoRA / QLoRA |
| Epochs | {NUM_EPOCHS} |
| Batch size | {BATCH_SIZE} |
| Gradient accumulation | {GRAD_ACCUM_STEPS} |
| Learning rate | {LEARNING_RATE} |
| Max length | {MAX_LENGTH} |
| LoRA rank | {LORA_R} |
| LoRA alpha | {LORA_ALPHA} |
| LoRA dropout | {LORA_DROPOUT} |
| Quantization | 4-bit NF4 |
| Precision | bfloat16 |

LoRA target modules:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## Results

Lower CER/WER is better.

| Run | System | Validation CER ↓ | Validation WER ↓ | Test CER ↓ | Test WER ↓ | Notes |
|---:|---|---:|---:|---:|---:|---|
| 0 | Original OCR | {baseline_cer} | {baseline_wer} | TBD | TBD | No correction |
| 1 | This adapter | TBD | TBD | TBD | TBD | Generation CER/WER not yet computed |

Loss-based validation metrics:

| Metric | Value |
|---|---:|
| Eval loss | {eval_loss} |
| Eval token accuracy | {eval_token_accuracy} |

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "{base_model}"
adapter_id = "{hub_model_id}"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

ocr_text = "GOOD TEMPLARS. At the quarterly meeting of tho Centennial Lodge..."

messages = [
    {{
        "role": "system",
        "content": (
            "You are an OCR post-correction system for historical newspaper text. "
            "Correct OCR transcription errors while preserving the original document as faithfully as possible. "
            "Return only the corrected transcription."
        ),
    }},
    {{
        "role": "user",
        "content": f"Correct the OCR transcription below.\\n\\nOCR text:\\n{{ocr_text}}",
    }},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print(prediction)
```

## Limitations

This adapter was trained on a small English historical newspaper dataset. It may overcorrect, hallucinate plausible text, or fail to preserve the source faithfully. Generation-based CER/WER should be computed before use in benchmark submissions or corpus processing.
"""

    readme_path = Path(output_dir) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme, encoding="utf-8")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
    )

    print(f"Uploaded model card to: https://huggingface.co/{hub_model_id}", flush=True)


def main():
    ds = load_from_disk(DATASET_PATH)

    print("=" * 100)
    print("DATASET")
    print(ds)
    print(f"Train examples: {len(ds['train'])}")
    print(f"Validation examples: {len(ds['validation'])}")
    print(f"Test examples: {len(ds['test'])}")
    print("=" * 100, flush=True)

    ocr_baseline_metrics = print_initial_ocr_baseline(ds["validation"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        print("=" * 100)
        print("GPU")
        print(torch.cuda.get_device_name(0))
        print(f"Memory allocated GB: {torch.cuda.memory_allocated() / 1024**3:.2f}")
        print(f"Memory reserved GB: {torch.cuda.memory_reserved() / 1024**3:.2f}")
        print("=" * 100, flush=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        max_length=MAX_LENGTH,
        packing=False,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora_config,
        args=args,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    print("TRAIN RESULT")
    print(train_result, flush=True)

    final_eval_metrics = trainer.evaluate()
    print("FINAL EVAL METRICS")
    print(final_eval_metrics, flush=True)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.push_to_hub()

    write_and_upload_model_card(
        hub_model_id=HUB_MODEL_ID,
        output_dir=OUTPUT_DIR,
        base_model=MODEL_NAME,
        train_examples=len(ds["train"]),
        validation_examples=len(ds["validation"]),
        test_examples=len(ds["test"]),
        ocr_baseline_metrics=ocr_baseline_metrics,
        final_eval_metrics=final_eval_metrics,
    )


if __name__ == "__main__":
    main()
