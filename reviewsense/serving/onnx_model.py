"""The star-rating model (Part 2) for serving: exported to ONNX and quantized to 8-bit integers (INT8).

  * ONNX is a standard file format for models; ONNX Runtime runs it without PyTorch, with a smaller install and
    graph optimizations. The exported model gives the same outputs as PyTorch.
  * Dynamic INT8 quantization stores the weights as 8-bit integers instead of 32-bit floats: 4x smaller.
Measured on 1,000 of Part 2's test reviews (Apple M-series CPU):
  pytorch 64.4% accuracy, 16.1 ms per review | onnx 64.4% (identical answers), 8.7 ms | onnx int8 65.2%, 8.6 ms, 67 MB
INT8 changed 7% of individual answers but not the accuracy (McNemar p = 0.40), so it is the served model. It was no
faster than ONNX here: ONNX Runtime's INT8 speed-ups mainly target x86 server CPUs, so measure again on the server.

  python -m reviewsense.serving.onnx_model export      # writes settings.star_onnx_dir
  python -m reviewsense.serving.onnx_model evaluate    # accuracy and speed: PyTorch vs ONNX vs ONNX INT8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings

STARS = np.arange(1, 6)


def export(model_dir: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    example = tokenizer(["The food was great but the service was slow."], return_tensors="pt")
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
    torch.onnx.export(model, (example["input_ids"], example["attention_mask"]), str(out_dir / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamo=True,
                      dynamic_shapes={"input_ids": {0: batch, 1: sequence}, "attention_mask": {0: batch, 1: sequence}})
    # The exporter stores shape annotations that the quantizer's shape check rejects ("Inferred shape and existing
    # shape differ"); they are only hints, so they are removed before quantizing.
    graph = onnx.load(str(out_dir / "model.onnx"))
    del graph.graph.value_info[:]
    onnx.save(graph, str(out_dir / "model-clean.onnx"), save_as_external_data=True, location="model-clean.onnx.data")
    quantize_dynamic(str(out_dir / "model-clean.onnx"), str(out_dir / "model-int8.onnx"), weight_type=QuantType.QInt8)
    for temporary in ["model-clean.onnx", "model-clean.onnx.data"]:
        (out_dir / temporary).unlink()
    tokenizer.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)                      # the label names ("1 star" ... "5 stars")


class OnnxStarModel:
    """Same answers as the Part 3 star model: expected stars (1.0-5.0) from the five class probabilities."""

    def __init__(self, folder: Path | None = None, int8: bool = True):
        folder = Path(folder or get_settings().star_onnx_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        model_file = "model-int8.onnx" if int8 else "model.onnx"
        self.session = onnxruntime.InferenceSession(str(folder / model_file), providers=["CPUExecutionProvider"])

    def probabilities(self, texts: list[str], max_length: int = 256) -> np.ndarray:
        inputs = self.tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="np")
        logits = self.session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})[0]
        return softmax(logits, axis=-1)

    def stars(self, texts: list[str]) -> list[float]:
        return [round(float(value), 2) for value in self.probabilities(texts) @ STARS]


class TorchStarModel:
    """The original PyTorch model, for comparison."""

    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    def probabilities(self, texts: list[str], max_length: int = 256) -> np.ndarray:
        inputs = self.tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            return softmax(self.model(**inputs).logits.numpy(), axis=-1)


def compare(n: int = 1000) -> dict:
    """Accuracy on Part 2's test reviews, agreement with PyTorch, and latency for one review (what an API call sees)."""
    from experiments.significance import compare as paired_test

    from ..data import load_hf_yelp

    test = load_hf_yelp("test", 2000).select(range(n))          # the same test reviews Part 2 scored
    texts, labels = list(test["text"]), np.array(test["label"])
    models = {"pytorch": TorchStarModel(get_settings().sentiment_model),
              "onnx": OnnxStarModel(int8=False), "onnx_int8": OnnxStarModel(int8=True)}
    results, correct = {}, {}
    for name, model in models.items():
        predictions = np.concatenate([model.probabilities(texts[i:i + 32]).argmax(-1) for i in range(0, n, 32)])
        correct[name] = predictions == labels
        timings = []
        for text in texts[:100]:
            start = time.perf_counter()
            model.probabilities([text])
            timings.append((time.perf_counter() - start) * 1000)
        results[name] = {"accuracy": round(float(correct[name].mean()), 4),
                         "latency_p50_ms": round(float(np.percentile(timings, 50)), 1),
                         "latency_p95_ms": round(float(np.percentile(timings, 95)), 1)}
    for name in ["onnx", "onnx_int8"]:
        results[name]["same_as_pytorch"] = round(float((correct[name] == correct["pytorch"]).mean()), 4)
        results[name]["mcnemar_p_vs_pytorch"] = paired_test(correct["pytorch"], correct[name])["mcnemar_p"]
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the star model to ONNX/INT8, or compare the versions.")
    parser.add_argument("command", choices=["export", "evaluate"])
    parser.add_argument("--n", type=int, default=1000, help="evaluate: number of test reviews")
    args = parser.parse_args(argv)
    if args.command == "export":
        export(get_settings().sentiment_model, get_settings().star_onnx_dir)
        print(f"wrote {get_settings().star_onnx_dir}")
    else:
        for name, scores in compare(args.n).items():
            print(f"{name:10} {scores}")


if __name__ == "__main__":
    main()
