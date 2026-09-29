"""
Benchmark & Objective Freezing for Quantum (Stage 6).
Locks immutable acceptance criteria to prevent moving the goalposts during evaluation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from quantum.constants import BASELINE_FILE, BENCHMARK_SPEC_FILE, QUANTUM_DIR


class ObjectiveFreezer:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)

    def freeze_objective(
        self,
        min_accuracy: float = 0.90,
        min_json_rate: float = 0.98,
        custom_metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates and signs the immutable benchmark specification.
        """
        # Load baseline for comparative reference
        baseline_file = self.root_dir / BASELINE_FILE
        baseline_data = {}
        if baseline_file.exists():
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
            except Exception:
                pass

        baseline_acc = baseline_data.get("metrics", {}).get("accuracy", 0.0)
        baseline_json = baseline_data.get("metrics", {}).get("valid_json_rate", 0.0)

        criteria = {
            "min_accuracy": round(min_accuracy, 4),
            "min_valid_json_rate": round(min_json_rate, 4),
            "baseline_accuracy_reference": baseline_acc,
            "baseline_json_rate_reference": baseline_json,
            "required_accuracy_gain": round(min_accuracy - baseline_acc, 4),
            "custom_metrics": custom_metrics or {},
        }

        spec_content = {
            "stage": "freeze-objective",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "acceptance_criteria": criteria,
            "status": "LOCKED",
            "notes": notes or "Critérios de aceitação obrigatórios para aprovação em produção.",
        }

        # Calculate integrity signature
        payload_bytes = json.dumps(criteria, sort_keys=True).encode("utf-8")
        signature = hashlib.sha256(payload_bytes).hexdigest()
        spec_content["spec_signature"] = signature

        out_path = self.root_dir / BENCHMARK_SPEC_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec_content, f, indent=2, ensure_ascii=False)

        return spec_content
