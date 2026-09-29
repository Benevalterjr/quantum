"""
Baseline Evaluation Runner for Quantum (Stage 5).
Measures base model zero-shot performance on reference test cases.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantum.constants import BASELINE_FILE, QUANTUM_DIR


class BaselineEvaluator:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)

    @staticmethod
    def is_valid_json(text: str) -> bool:
        """Check if output contains a parseable JSON block."""
        text = text.strip()
        # Direct json parse
        try:
            json.loads(text)
            return True
        except Exception:
            pass
        # Markdown fence parse
        match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return True
            except Exception:
                pass
        return False

    def run_baseline(
        self,
        eval_set_path: Path | str,
        backend: str = "mock",
        base_model_name: str = "Base Model",
    ) -> Dict[str, Any]:
        """
        Executes zero-shot baseline assessment across eval set.
        """
        eval_p = Path(eval_set_path)
        if not eval_p.is_absolute():
            eval_p = (self.root_dir / eval_p).resolve()

        if not eval_p.exists():
            raise FileNotFoundError(f"Arquivo de avaliação baseline não encontrado: {eval_p}")

        # Load samples
        samples: List[Dict[str, Any]] = []
        if eval_p.suffix == ".jsonl":
            with open(eval_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(eval_p, "r", encoding="utf-8") as f:
                samples = json.load(f)

        total_samples = len(samples)
        if total_samples == 0:
            raise ValueError("O dataset de avaliação está vazio.")

        results: List[Dict[str, Any]] = []
        valid_json_count = 0
        correct_count = 0

        # Execute evaluation per sample
        for idx, sample in enumerate(samples, start=1):
            if backend == "mock":
                # Simulated zero-shot base model behavior:
                # Base models often hallucinate extra markdown, miss fields, or fail domain specifics
                # Typically ~40% accuracy and ~60% valid json zero-shot
                is_json_valid = (idx % 3 != 0)  # ~66% valid json
                is_correct = (idx % 2 == 0 and is_json_valid)  # ~33% correct
                pred_text = '{"status": "ok"}' if is_json_valid else "Desculpe, como modelo de linguagem..."
            else:
                # TODO: Optional live Ollama / API call
                pred_text = "Simulated output"
                is_json_valid = self.is_valid_json(pred_text)
                is_correct = False

            if is_json_valid:
                valid_json_count += 1
            if is_correct:
                correct_count += 1

            results.append({
                "test_id": idx,
                "is_valid_json": is_json_valid,
                "is_correct": is_correct,
            })

        valid_json_rate = round(valid_json_count / total_samples, 4)
        accuracy_score = round(correct_count / total_samples, 4)

        report = {
            "stage": "baseline",
            "base_model": base_model_name,
            "backend": backend,
            "eval_dataset": eval_p.name,
            "total_test_cases": total_samples,
            "metrics": {
                "accuracy": accuracy_score,
                "valid_json_rate": valid_json_rate,
                "format_compliance_rate": valid_json_rate,
            },
            "summary": (
                f"Régua Inicial Zero-Shot: Acurácia={accuracy_score*100:.1f}%, "
                f"JSON Válido={valid_json_rate*100:.1f}% em {total_samples} amostras."
            ),
        }

        # Persist to .quantum/baseline_results.json
        out_path = self.root_dir / BASELINE_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report
