"""
Tests for Quantum Guardrails and Data Privacy Preserving Mechanism.
"""
import json
import pytest
from pathlib import Path

from quantum.guardrails.schema_validator import SchemaValidator
from quantum.guardrails.token_counter import LocalTokenCounter
from quantum.guardrails.anonymizer import DataCheckGuardrail


def test_schema_validator_alpaca_valid(tmp_path):
    f_path = tmp_path / "train.jsonl"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "Classifique o texto", "input": "Texto 1", "output": "Positivo"}) + "\n")
        f.write(json.dumps({"instruction": "Extraia dados", "output": '{"chave": "valor"}'}) + "\n")

    res = SchemaValidator.validate_file(f_path, schema="alpaca")
    assert res["is_valid"] is True
    assert res["total_records"] == 2
    assert res["valid_records"] == 2
    assert res["invalid_records"] == 0
    assert len(res["errors"]) == 0


def test_schema_validator_missing_fields(tmp_path):
    f_path = tmp_path / "bad.jsonl"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "Texto sem output"}) + "\n")
        f.write(json.dumps({"output": "Output sem instruction"}) + "\n")

    res = SchemaValidator.validate_file(f_path, schema="alpaca")
    assert res["is_valid"] is False
    assert res["total_records"] == 2
    assert res["invalid_records"] == 2
    assert "campo 'output' ausente" in res["errors"][0]
    assert "campo 'instruction' ausente" in res["errors"][1]


def test_schema_validator_chatml(tmp_path):
    f_path = tmp_path / "chatml.jsonl"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "messages": [
                {"role": "system", "content": "Voce e um assistente."},
                {"role": "user", "content": "Ola mundo."},
                {"role": "assistant", "content": "Ola! Como posso ajudar?"},
            ]
        }) + "\n")

    res = SchemaValidator.validate_file(f_path, schema="chatml")
    assert res["is_valid"] is True
    assert res["detected_schema"] == "chatml"
    assert res["valid_records"] == 1


def test_local_token_counter():
    tc = LocalTokenCounter()
    texts = [
        "Palavra simples",
        "Este é um texto um pouco maior com várias palavras para contar tokens.",
        " ".join(["palavra"] * 300),
    ]
    dist = tc.compute_distribution(texts)
    assert dist["count"] == 3
    assert dist["min"] > 0
    assert dist["max"] >= 300
    assert dist["p50"] > 0
    assert dist["p99"] >= dist["p95"]
    assert dist["recommended_max_seq_length"] in (512, 1024, 2048, 4096, 8192)


def test_privacy_guardrail_zero_raw_data_leakage(tmp_path):
    """
    CRITICAL PRIVACY TEST:
    Ensures NO secret string or PII from the raw training dataset is written
    to .quantum/telemetry.json or returned in telemetry summary.
    """
    secret_text = "SECRET_CORPORATE_FINANCIAL_RECORD_XYZ_987654321"
    secret_prompt = "CONFIDENTIAL_PATIENT_MEDICAL_HISTORY_DIAGNOSIS_42"

    dataset_file = tmp_path / "confidential_data.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({
                "instruction": f"{secret_prompt} - Exemplo #{i}",
                "input": f"Dados sigilosos #{i}",
                "output": f"{secret_text} resultado #{i}"
            }) + "\n")

    guardrail = DataCheckGuardrail(root_dir=tmp_path)
    telemetry = guardrail.inspect_dataset(dataset_file, schema="alpaca")

    # Serialize telemetry to text to search for any leak
    telemetry_dump = json.dumps(telemetry)

    # 1. Assert telemetry status is passed
    assert telemetry["guardrail_status"] == "PASSED"
    assert telemetry["privacy_guarantee"] == "ZERO_RAW_DATA_LEAKED_TO_LLM"
    assert telemetry["dataset_info"]["valid_records"] == 10

    # 2. Assert secrets are ABSENT from telemetry
    assert secret_text not in telemetry_dump
    assert secret_prompt not in telemetry_dump
    assert "Dados sigilosos" not in telemetry_dump

    # 3. Assert file on disk is also completely sanitized
    saved_telemetry_file = tmp_path / ".quantum" / "telemetry.json"
    assert saved_telemetry_file.exists()
    disk_content = saved_telemetry_file.read_text(encoding="utf-8")
    assert secret_text not in disk_content
    assert secret_prompt not in disk_content
