"""
Data Check Guardrail & Telemetry Generator for Quantum.
Ensures 100% of data inspection happens locally and strictly prevents PII leakage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from quantum.constants import QUANTUM_DIR, TELEMETRY_FILE
from quantum.guardrails.schema_validator import SchemaValidator
from quantum.guardrails.token_counter import LocalTokenCounter


class DataCheckGuardrail:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)
        self.token_counter = LocalTokenCounter()

    @staticmethod
    def calculate_file_hash(path: Path) -> str:
        """Calculates SHA256 of file for provenance and integrity."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect_dataset(
        self,
        dataset_path: Path | str,
        val_dataset_path: Optional[Path | str] = None,
        schema: str = "auto",
    ) -> Dict[str, Any]:
        """
        Inspects training (and optional validation) datasets.
        Returns ONLY sanitized aggregate metrics and writes telemetry.json.
        """
        train_p = Path(dataset_path)
        if not train_p.is_absolute():
            train_p = (self.root_dir / train_p).resolve()

        if not train_p.exists():
            raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_p}")

        file_size_bytes = os.path.getsize(train_p)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 3)
        file_sha256 = self.calculate_file_hash(train_p)

        # 1. Validate Schema
        validation_res = SchemaValidator.validate_file(train_p, schema=schema)
        internal_texts = validation_res.pop("_internal_texts", [])

        # 2. Compute Token Statistics
        token_stats = self.token_counter.compute_distribution(internal_texts)

        # 3. Optional Validation Set
        val_summary = None
        if val_dataset_path:
            val_p = Path(val_dataset_path)
            if not val_p.is_absolute():
                val_p = (self.root_dir / val_p).resolve()
            if val_p.exists():
                v_res = SchemaValidator.validate_file(val_p, schema=schema)
                v_texts = v_res.pop("_internal_texts", [])
                v_toks = self.token_counter.compute_distribution(v_texts)
                val_summary = {
                    "file_name": val_p.name,
                    "file_size_mb": round(os.path.getsize(val_p) / (1024 * 1024), 3),
                    "total_records": v_res["total_records"],
                    "valid_records": v_res["valid_records"],
                    "invalid_records": v_res["invalid_records"],
                    "token_stats": {
                        "min": v_toks["min"],
                        "mean": v_toks["mean"],
                        "max": v_toks["max"],
                        "p95": v_toks["p95"],
                    },
                }

        # Clean telemetry payload - STRICTLY NO RAW DATA OR PROMPTS
        telemetry = {
            "guardrail_status": "PASSED" if validation_res["is_valid"] else "FAILED_VALIDATION",
            "privacy_guarantee": "ZERO_RAW_DATA_LEAKED_TO_LLM",
            "dataset_info": {
                "file_name": train_p.name,
                "file_size_mb": file_size_mb,
                "sha256": file_sha256,
                "detected_schema": validation_res["detected_schema"],
                "total_records": validation_res["total_records"],
                "valid_records": validation_res["valid_records"],
                "invalid_records": validation_res["invalid_records"],
                "is_valid": validation_res["is_valid"],
                "errors": validation_res["errors"],
            },
            "token_distribution": token_stats,
            "recommended_max_seq_length": token_stats["recommended_max_seq_length"],
            "validation_dataset": val_summary,
            "training_feasibility": {
                "samples_sufficient": validation_res["valid_records"] >= 20,
                "token_cutoff_risk": token_stats["max"] > 4096,
                "estimated_steps_epoch_bs1": validation_res["valid_records"],
            },
        }

        # Persist to .quantum/telemetry.json
        out_path = self.root_dir / TELEMETRY_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)

        return telemetry
