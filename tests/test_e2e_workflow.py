"""
End-to-End Workflow Integration Test for the Quantum 11-Stage Harness.
Simulates a complete real-world fine-tuning journey through all 11 stages.
"""
import json
import pytest
from pathlib import Path
from typer.testing import CliRunner

from quantum.cli import app
from quantum.state_machine import QuantumStateMachine
from quantum.constants import (
    INTENT_FILE,
    ENV_FILE,
    TELEMETRY_FILE,
    MODEL_CHOICE_FILE,
    BASELINE_FILE,
    BENCHMARK_SPEC_FILE,
    RECIPE_FILE,
    DRY_RUN_FILE,
    TRAIN_LOG_FILE,
    EVAL_REPORT_FILE,
    EVAL_MARKDOWN_FILE,
    EXPORT_SPEC_FILE,
    DELIVERY_SPEC_FILE,
    MODELFILE_PATH,
    HANDOVER_FILE,
    TRAIN_SCRIPT_FILE,
    COLAB_NOTEBOOK_FILE,
)


def test_full_11_stage_workflow_e2e(tmp_path, monkeypatch):
    # Change working directory to tmp_path for isolated CLI execution
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # 0. Quantum Init
    res = runner.invoke(app, ["init", "--name", "e2e-suporte-juridico"])
    assert res.exit_code == 0
    assert "Inicializado com Sucesso" in res.output

    # 1. Stage 1: Intenção
    res = runner.invoke(app, [
        "stage", "intent",
        "--task", "extraction",
        "--domain", "Direito Tributario",
        "--output-format", "json",
        "--description", "Extracao de partes, CNPJ e valores em peticoes"
    ])
    assert res.exit_code == 0
    assert (tmp_path / INTENT_FILE).exists()

    # 2. Stage 2: Ambiente
    res = runner.invoke(app, [
        "stage", "env",
        "--target", "colab",
        "--profile", "colab-free",
    ])
    assert res.exit_code == 0
    assert (tmp_path / ENV_FILE).exists()

    # Create dummy datasets for Stage 3 & 5
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for i in range(25):
            f.write(json.dumps({
                "instruction": f"Extraia os dados da peticao #{i}",
                "input": f"Processo #{1000+i} da comarca tributaria",
                "output": json.dumps({"processo": 1000+i, "status": "deferido"})
            }) + "\n")

    with open(eval_file, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(json.dumps({
                "instruction": f"Teste de extracao #{i}",
                "output": json.dumps({"resultado": "ok"})
            }) + "\n")

    # 3. Stage 3: Dados & Guardrails
    res = runner.invoke(app, [
        "stage", "data-check",
        "--dataset", str(train_file),
        "--schema", "alpaca"
    ])
    assert res.exit_code == 0
    assert (tmp_path / TELEMETRY_FILE).exists()

    # 4. Stage 4: Escolha do Modelo
    res = runner.invoke(app, ["stage", "model-select"])
    assert res.exit_code == 0
    assert (tmp_path / MODEL_CHOICE_FILE).exists()

    # 5. Stage 5: Régua Baseline
    res = runner.invoke(app, [
        "stage", "baseline",
        "--eval-set", str(eval_file),
        "--backend", "mock"
    ])
    assert res.exit_code == 0
    assert (tmp_path / BASELINE_FILE).exists()

    # 6. Stage 6: Congelar Objetivo
    res = runner.invoke(app, [
        "stage", "freeze-objective",
        "--min-accuracy", "0.92",
        "--min-json-rate", "0.98",
        "--notes", "Meta contratual para lancamento em producao"
    ])
    assert res.exit_code == 0
    assert (tmp_path / BENCHMARK_SPEC_FILE).exists()

    # 7. Stage 7: Receita
    res = runner.invoke(app, [
        "stage", "recipe",
        "--framework", "unsloth",
        "--quantization", "4bit",
        "--epochs", "2"
    ])
    assert res.exit_code == 0
    assert (tmp_path / RECIPE_FILE).exists()
    assert (tmp_path / TRAIN_SCRIPT_FILE).exists()
    assert (tmp_path / COLAB_NOTEBOOK_FILE).exists()

    # 8. Stage 8: Ensaio (Dry-Run) & Treinamento
    res = runner.invoke(app, ["stage", "dry-run", "--batches", "3"])
    assert res.exit_code == 0
    assert (tmp_path / DRY_RUN_FILE).exists()

    res = runner.invoke(app, ["stage", "train", "--mock", "--steps", "30"])
    assert res.exit_code == 0
    assert (tmp_path / TRAIN_LOG_FILE).exists()

    # 9. Stage 9: Avaliação
    res = runner.invoke(app, ["stage", "evaluate", "--mock-accuracy", "0.96", "--mock-json-rate", "1.0"])
    assert res.exit_code == 0
    assert (tmp_path / EVAL_REPORT_FILE).exists()
    assert (tmp_path / EVAL_MARKDOWN_FILE).exists()

    # 10. Stage 10: Exportação
    res = runner.invoke(app, ["stage", "export", "--format", "gguf", "--quant", "q4_k_m"])
    assert res.exit_code == 0
    assert (tmp_path / EXPORT_SPEC_FILE).exists()

    # 11. Stage 11: Entrega
    res = runner.invoke(app, ["stage", "deliver", "--target", "ollama", "--tag", "quantum-tributario:v1"])
    assert res.exit_code == 0
    assert (tmp_path / DELIVERY_SPEC_FILE).exists()
    assert (tmp_path / MODELFILE_PATH).exists()
    assert (tmp_path / HANDOVER_FILE).exists()

    # Verify final state machine status
    sm = QuantumStateMachine(root_dir=tmp_path)
    for s_id in range(1, 12):
        assert sm.is_stage_completed(s_id) is True, f"Etapa {s_id} deveria estar concluida!"

    # Verify status table output
    res_status = runner.invoke(app, ["status"])
    assert res_status.exit_code == 0
    assert "[OK] Concluido" in res_status.output
