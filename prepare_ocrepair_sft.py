import json
from pathlib import Path

from datasets import Dataset, DatasetDict

SYSTEM_PROMPT = (
    "You are an OCR post-correction system for historical newspaper text. "
    "Your task is to correct OCR transcription errors while preserving the original document as faithfully as possible. "
    "Use the metadata only as contextual information about the document, such as language, date, publication, and document type. "
    "The OCR text is the source to correct. Do not add information that is not supported by the OCR text. "
    "Do not summarize, paraphrase, translate, modernize, normalize, or rewrite the text. "
    "Preserve historical spelling, historical vocabulary, names, dates, abbreviations, punctuation, capitalization, and line breaks whenever possible. "
    "Return only the corrected transcription, with no explanation."
)


def read_jsonl(path: Path) -> list[dict]:
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
                    "primary_dataset_name": meta.get("primary_dataset_name"),
                    "primary_dataset_version": meta.get("primary_dataset_version"),
                    "document_type": meta.get("document_type"),
                    "date": meta.get("date"),
                    "language": meta.get("language"),
                    "publication_title": meta.get("publication_title"),
                    "transcription_unit_scope": meta.get("transcription_unit_scope"),
                    "segmentation_origin_article": meta.get(
                        "segmentation_origin_article"
                    ),
                    "segmentation_origin_lines": meta.get("segmentation_origin_lines"),
                    "segmentation_origin_sentences": meta.get(
                        "segmentation_origin_sentences"
                    ),
                }
            )

    return rows


def clean_value(value) -> str:
    if value is None:
        return "unknown"
    if value == "":
        return "unknown"
    return str(value)


def build_metadata_block(example: dict) -> str:
    """
    Build a compact metadata block using only information that would be
    available at inference time.

    Do NOT include OCR CER/WER, OCR quality scores, edit distances, or any
    field derived from comparison with the ground truth.
    """

    metadata = {
        "document id": example.get("document_id"),
        "dataset": example.get("primary_dataset_name"),
        "dataset version": example.get("primary_dataset_version"),
        "document type": example.get("document_type"),
        "language": example.get("language"),
        "date": example.get("date"),
        "publication title": example.get("publication_title"),
        "transcription unit": example.get("transcription_unit_scope"),
        "article segmentation": example.get("segmentation_origin_article"),
        "line segmentation": example.get("segmentation_origin_lines"),
        "sentence segmentation": example.get("segmentation_origin_sentences"),
    }

    lines = []

    for key, value in metadata.items():
        value = clean_value(value)
        if value != "unknown":
            lines.append(f"- {key}: {value}")

    if not lines:
        return "No metadata available."

    return "\n".join(lines)


def to_chat_example(example: dict) -> dict:
    metadata_block = build_metadata_block(example)

    user_prompt = (
        "Correct the OCR transcription below.\n\n"
        "Document metadata:\n"
        f"{metadata_block}\n\n"
        "Correction objective:\n"
        "Produce a faithful post-OCR correction of the text. The goal is not to improve style, "
        "but to recover the text that was most likely present in the original historical newspaper.\n\n"
        "Rules:\n"
        "1. Correct OCR errors in words, characters, punctuation, spacing, and line-break artefacts.\n"
        "2. Preserve the original meaning and wording. Do not paraphrase.\n"
        "3. Preserve historical spelling and historical vocabulary. Do not modernize.\n"
        "4. Preserve named entities, titles, dates, numbers, abbreviations, initials, and place names as accurately as possible.\n"
        "5. Preserve capitalization and punctuation when they appear intentional in the source.\n"
        "6. Preserve line breaks as much as possible. Only change line breaks when needed to fix clear OCR or hyphenation errors.\n"
        "7. Resolve obvious OCR confusions, such as mistaken letters, digits, accents, broken words, or corrupted punctuation.\n"
        "8. Do not add missing sentences or facts. If the OCR is ambiguous, make the smallest plausible correction.\n"
        "9. Do not output comments, explanations, markdown, or metadata.\n"
        "10. Return only the corrected transcription.\n\n"
        "OCR text:\n"
        f"{example['ocr']}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["ground_truth"]},
    ]

    return {
        **example,
        "metadata_prompt": metadata_block,
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
        rows = [to_chat_example(row) for row in rows]
        ds[split] = Dataset.from_list(rows)

    ds.save_to_disk("overproof_ocrepair_sft")

    print(ds)
    print("=" * 100)
    print("Example metadata block")
    print("=" * 100)
    print(ds["train"][0]["metadata_prompt"])
    print("=" * 100)
    print("Example chat messages")
    print("=" * 100)
    print(ds["train"][0]["messages"])


if __name__ == "__main__":
    main()
