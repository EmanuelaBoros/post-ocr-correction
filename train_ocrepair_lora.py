import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "overproof_ocrepair_sft"
OUTPUT_DIR = "qwen2_5_3b_overproof_ocrepair_lora"

def formatting_func(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

def main():
    ds = load_from_disk(DATASET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        max_seq_length=4096,
        packing=False,
        report_to="none",
        push_to_hub=True,
        hub_model_id="emanuelaboros/qwen2-5-3b-overproof-postcorrection",
        hub_private_repo=True,  # or False if you want it public
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora_config,
        args=args,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()