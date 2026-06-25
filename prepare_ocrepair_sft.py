import json
from pathlib import Path
from datasets import Dataset, DatasetDict

SYSTEM_PROMPT = (
    "You are an OCR post-correction system for historical English newspapers. "
    "Correct OCR errors while preserving the original wording, punctuation, line breaks, "
    "spelling style, names, dates, and historical language. Do not modernize the text. "
    "Return only the corrected text."
)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            ocr = obj["ocr_hypothesis"]["transcription_unit"]
            gt = obj["ground_truth"]["transcription_unit"]
            meta = obj.get("document_metadata", {})

            if not ocr or not gt:
                continue

            if ocr == "None" or gt == "None":
                continue

            rows.append(
                {
                    "ocr": ocr,
                    "ground_truth": gt,
                    "document_id": meta.get("document_id"),
                    "date": meta.get("date"),
                    "language": meta.get("language"),
                    "publication_title": meta.get("publication_title"),
                }
            )

    return rows


def to_chat_example(example):
    user = (
        "Correct the following OCR text from a historical newspaper.\n\n"
        f"OCR text:\n{example['ocr']}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["ground_truth"]},
    ]

    return {
        **example,
        "messages": messages,
    }


def main():
    base = Path("data")

    files = {
        "train": base
        / "hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_train_en.jsonl",
        "validation": base
        / "hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_dev_en.jsonl",
        "test": base / "hipe-ocrepair-bench_v0.9_overproof-combined_v1.0_test_en.jsonl",
    }

    ds = DatasetDict()

    for split, path in files.items():
        rows = read_jsonl(path)
        rows = [to_chat_example(x) for x in rows]
        ds[split] = Dataset.from_list(rows)

    ds.save_to_disk("overproof_ocrepair_sft")
    print(ds)
    print(ds["train"][0]["messages"])


if __name__ == "__main__":
    main()
