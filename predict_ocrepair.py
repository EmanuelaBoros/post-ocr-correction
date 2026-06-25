import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "qwen2_5_3b_overproof_ocrepair_lora"
INPUT_FILE = "data/v0.9/overproof/en/hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_test_en.jsonl"
OUTPUT_FILE = "predictions_overproof_test.jsonl"

SYSTEM_PROMPT = (
    "You are an OCR post-correction system for historical English newspapers. "
    "Correct OCR errors while preserving the original wording, punctuation, line breaks, "
    "spelling style, names, dates, and historical language. Do not modernize the text. "
    "Return only the corrected text."
)

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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            ocr = obj["ocr_hypothesis"]["transcription_unit"]

            prompt = build_prompt(tokenizer, ocr)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.05,
                )

            generated = output_ids[0][inputs["input_ids"].shape[-1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

            obj["ocr_postcorrection_output"] = {
                "transcription_unit": prediction,
                "ocr_postcorrection_system": "qwen2.5-3b-lora-overproof",
            }

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()