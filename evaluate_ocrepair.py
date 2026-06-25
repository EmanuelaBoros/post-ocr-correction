import json
from jiwer import wer
import evaluate

cer_metric = evaluate.load("cer")

refs = []
preds = []

with open("predictions_overproof_test.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        refs.append(obj["ground_truth"]["transcription_unit"])
        preds.append(obj["ocr_postcorrection_output"]["transcription_unit"])

print("CER:", cer_metric.compute(predictions=preds, references=refs))
print("WER:", wer(refs, preds))