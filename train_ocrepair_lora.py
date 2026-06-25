import torch
import evaluate
from jiwer import wer
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "overproof_ocrepair_sft"
OUTPUT_DIR = "qwen2_5_3b_overproof_ocrepair_lora"
HUB_MODEL_ID = "emanuelaboros/qwen2-5-3b-overproof-postcorrection"

# For fast evaluation during training.
# Increase later if you want more stable numbers.
CER_WER_EVERY_N_STEPS = 100
CER_WER_MAX_EVAL_SAMPLES = 50
CER_WER_MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are an OCR post-correction system for historical English newspapers. "
    "Correct OCR errors while preserving the original wording, punctuation, line breaks, "
    "spelling style, names, dates, and historical language. Do not modernize the text. "
    "Return only the corrected text."
)

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


def print_initial_ocr_baseline(eval_dataset, max_samples=None):
    if max_samples is not None:
        n = min(max_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))

    ocr_texts = list(eval_dataset["ocr"])
    references = list(eval_dataset["ground_truth"])

    cer, word_error_rate = compute_cer_wer(ocr_texts, references)

    print("=" * 100)
    print("INITIAL OCR BASELINE ON VALIDATION SET")
    print(f"Samples: {len(eval_dataset)}")
    print(f"OCR CER: {cer:.6f}")
    print(f"OCR WER: {word_error_rate:.6f}")
    print("=" * 100, flush=True)


class OCRCorrectionEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        every_n_steps=100,
        max_eval_samples=50,
        max_new_tokens=1024,
    ):
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens

        if max_eval_samples is not None:
            n = min(max_eval_samples, len(eval_dataset))
            self.eval_dataset = eval_dataset.select(range(n))
        else:
            self.eval_dataset = eval_dataset

    def build_prompt(self, ocr_text):
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

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        if state.global_step == 0:
            return control

        if state.global_step % self.every_n_steps != 0:
            return control

        print("=" * 100)
        print(f"GENERATIVE CER/WER EVALUATION AT STEP {state.global_step}")
        print(f"Samples: {len(self.eval_dataset)}")
        print("=" * 100, flush=True)

        model.eval()

        predictions = []
        references = []

        for i, example in enumerate(self.eval_dataset):
            prompt = self.build_prompt(example["ocr"])

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            prediction = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            predictions.append(prediction)
            references.append(example["ground_truth"])

        cer, word_error_rate = compute_cer_wer(predictions, references)

        print(f"Step {state.global_step} generated CER: {cer:.6f}")
        print(f"Step {state.global_step} generated WER: {word_error_rate:.6f}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            print(f"GPU memory allocated GB: {allocated:.2f}")
            print(f"GPU memory reserved GB: {reserved:.2f}")
            print(f"GPU max memory allocated GB: {max_allocated:.2f}")

        print("-" * 100)
        print("Example OCR:")
        print(self.eval_dataset[0]["ocr"][:700])
        print("\nExample prediction:")
        print(predictions[0][:700])
        print("\nExample reference:")
        print(references[0][:700])
        print("=" * 100, flush=True)

        model.train()
        return control


def main():
    ds = load_from_disk(DATASET_PATH)

    print("=" * 100)
    print("DATASET")
    print(ds)
    print(f"Train examples: {len(ds['train'])}")
    print(f"Validation examples: {len(ds['validation'])}")
    print("=" * 100, flush=True)

    print_initial_ocr_baseline(
        ds["validation"],
        max_samples=None,  # full validation baseline; change to 100 if slow
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Proper 4-bit quantization config for newer Transformers.
    # This avoids the previous load_in_4bit TypeError.
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        # More visible logs.
        logging_steps=10,
        # Standard eval loss.
        eval_strategy="steps",
        eval_steps=100,
        # Checkpoints.
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        # 4096 may be okay on A10G with 4-bit.
        # If OOM, reduce to 2048.
        max_length=4096,
        packing=False,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora_config,
        args=args,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        callbacks=[
            OCRCorrectionEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=ds["validation"],
                every_n_steps=CER_WER_EVERY_N_STEPS,
                max_eval_samples=CER_WER_MAX_EVAL_SAMPLES,
                max_new_tokens=CER_WER_MAX_NEW_TOKENS,
            )
        ],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
