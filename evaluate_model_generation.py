import argparse
import json
from pathlib import Path

import torch
import evaluate
from jiwer import wer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an OCR post-correction system for historical English newspapers. "
    "Correct OCR errors while preserving the original wording, punctuation, line breaks, "
    "spelling style, names, dates, and historical language. Do not modernize the text. "
    "Return only the corrected text."
)

cer_metric = evaluate.load("cer")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_texts(record):
    ocr = record["ocr_hypothesis"]["transcription_unit"]
    gt = record["ground_truth"]["transcription_unit"]
    return ocr, gt


def compute_cer_wer(predictions, references):
    predictions = [str(x) if x is not None else "" for x in predictions]
    references = [str(x) if x is not None else "" for x in references]

    cer = cer_metric.compute(predictions=predictions, references=references)
    word_error_rate = wer(references, predictions)

    return cer, word_error_rate


def build_prompt(tokenizer, ocr_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Correct the following OCR text from a historical newspaper.\n\n"
                f"OCR text:\n{ocr_text}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--adapter",
        default="emanuelaboros/qwen2-5-3b-overproof-postcorrection",
    )
    parser.add_argument(
        "--input_file",
        default="data/hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_dev_en.jsonl",
    )
    parser.add_argument(
        "--output_file",
        default="predictions_overproof_dev.jsonl",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load the base model in 4-bit quantization.",
    )

    args = parser.parse_args()

    records = load_jsonl(args.input_file)

    if args.limit is not None:
        records = records[: args.limit]

    print("=" * 100)
    print("GENERATION EVALUATION")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Examples: {len(records)}")
    print("=" * 100, flush=True)

    ocr_texts = []
    references = []

    for record in records:
        ocr, gt = get_texts(record)
        ocr_texts.append(ocr)
        references.append(gt)

    baseline_cer, baseline_wer = compute_cer_wer(ocr_texts, references)

    print("=" * 100)
    print("OCR BASELINE")
    print(f"CER: {baseline_cer:.6f}")
    print(f"WER: {baseline_wer:.6f}")
    print("=" * 100, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    if torch.cuda.is_available():
        print("=" * 100)
        print("GPU")
        print(torch.cuda.get_device_name(0))
        print(f"Memory allocated GB: {torch.cuda.memory_allocated() / 1024**3:.2f}")
        print(f"Memory reserved GB: {torch.cuda.memory_reserved() / 1024**3:.2f}")
        print("=" * 100, flush=True)

    predictions = []

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records, desc="Generating"):
            ocr, gt = get_texts(record)

            prompt = build_prompt(tokenizer, ocr)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            predictions.append(prediction)

            record["ocr_postcorrection_output"] = {
                "transcription_unit": prediction,
                "ocr_postcorrection_system": args.adapter,
                "num_tokens": len(tokenizer.encode(prediction)),
                "num_chars": len(prediction),
                "quality_report": {},
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    model_cer, model_wer = compute_cer_wer(predictions, references)

    print("=" * 100)
    print("MODEL RESULTS")
    print(f"CER: {model_cer:.6f}")
    print(f"WER: {model_wer:.6f}")
    print("=" * 100)

    print("IMPROVEMENT OVER OCR BASELINE")
    print(f"CER absolute change: {model_cer - baseline_cer:+.6f}")
    print(f"WER absolute change: {model_wer - baseline_wer:+.6f}")
    print(f"CER relative change: {(model_cer - baseline_cer) / baseline_cer:+.2%}")
    print(f"WER relative change: {(model_wer - baseline_wer) / baseline_wer:+.2%}")
    print("=" * 100)

    print("Example")
    print("-" * 100)
    print("OCR:")
    print(ocr_texts[0][:1000])
    print("-" * 100)
    print("Prediction:")
    print(predictions[0][:1000])
    print("-" * 100)
    print("Reference:")
    print(references[0][:1000])
    print("=" * 100)


if __name__ == "__main__":
    main()