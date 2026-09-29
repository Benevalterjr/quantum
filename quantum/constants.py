"""
Constants and definitions for the Quantum Framework.
"""
from pathlib import Path

# Quantum directory structure
QUANTUM_DIR = Path(".quantum")
STATE_FILE = QUANTUM_DIR / "state.json"
INTENT_FILE = QUANTUM_DIR / "intent.json"
ENV_FILE = QUANTUM_DIR / "environment.json"
TELEMETRY_FILE = QUANTUM_DIR / "telemetry.json"
MODEL_CHOICE_FILE = QUANTUM_DIR / "model_choice.json"
BASELINE_FILE = QUANTUM_DIR / "baseline_results.json"
BENCHMARK_SPEC_FILE = QUANTUM_DIR / "benchmark_spec.json"
RECIPE_FILE = QUANTUM_DIR / "recipe.json"
DRY_RUN_FILE = QUANTUM_DIR / "dry_run.json"
TRAIN_LOG_FILE = QUANTUM_DIR / "training_log.json"
EVAL_REPORT_FILE = QUANTUM_DIR / "evaluation_report.json"
EVAL_MARKDOWN_FILE = Path("evaluation_report.md")
EXPORT_SPEC_FILE = QUANTUM_DIR / "export_spec.json"
DELIVERY_SPEC_FILE = QUANTUM_DIR / "delivery_spec.json"

TRAIN_SCRIPT_FILE = Path("train.py")
COLAB_NOTEBOOK_FILE = Path("colab_train.ipynb")
MODELFILE_PATH = Path("Modelfile")
HANDOVER_FILE = Path("HANDOVER.md")
CHECKPOINTS_DIR = Path("checkpoints")
EXPORT_DIR = Path("export")

# The 11 Stages of Quantum
STAGES = [
    {
        "id": 1,
        "name": "intent",
        "title": "1. Intenção",
        "description": "Entrevista técnica e definição de tarefa, domínio e formato esperado",
        "artifact": str(INTENT_FILE),
    },
    {
        "id": 2,
        "name": "env",
        "title": "2. Ambiente",
        "description": "Detecção de hardware (VRAM, CUDA, CPU, RAM) ou perfil de nuvem",
        "artifact": str(ENV_FILE),
    },
    {
        "id": 3,
        "name": "data-check",
        "title": "3. Dados & Guardrails",
        "description": "Validação de dados e geração de telemetria estritamente anônima (Zero PII)",
        "artifact": str(TELEMETRY_FILE),
    },
    {
        "id": 4,
        "name": "model-select",
        "title": "4. Escolha do Modelo",
        "description": "Recomendação e seleção do foundation model ideal para a tarefa e VRAM",
        "artifact": str(MODEL_CHOICE_FILE),
    },
    {
        "id": 5,
        "name": "baseline",
        "title": "5. Régua Baseline",
        "description": "Execução de testes zero-shot para aferir métricas iniciais de referência",
        "artifact": str(BASELINE_FILE),
    },
    {
        "id": 6,
        "name": "freeze-objective",
        "title": "6. Congelar Objetivo",
        "description": "Fixação imutável dos critérios de aceitação e metas mínimas",
        "artifact": str(BENCHMARK_SPEC_FILE),
    },
    {
        "id": 7,
        "name": "recipe",
        "title": "7. Receita",
        "description": "Geração do script de treino independente (LoRA/QLoRA) e notebook Colab",
        "artifact": str(RECIPE_FILE),
    },
    {
        "id": 8,
        "name": "train",
        "title": "8. Ensaio & Treino",
        "description": "Ensaio preventivo de 2-5 batches e execução segura do treinamento",
        "artifact": str(TRAIN_LOG_FILE),
    },
    {
        "id": 9,
        "name": "evaluate",
        "title": "9. Avaliação",
        "description": "Reavaliação na mesma régua e validação contra o objetivo congelado",
        "artifact": str(EVAL_REPORT_FILE),
    },
    {
        "id": 10,
        "name": "export",
        "title": "10. Exportação",
        "description": "Merge LoRA para Safetensors e quantização GGUF para inferência",
        "artifact": str(EXPORT_SPEC_FILE),
    },
    {
        "id": 11,
        "name": "deliver",
        "title": "11. Entrega",
        "description": "Geração do Modelfile (Ollama), vLLM launcher e documentação técnica",
        "artifact": str(DELIVERY_SPEC_FILE),
    },
]

STAGE_NAME_TO_ID = {stage["name"]: stage["id"] for stage in STAGES}
STAGE_ID_TO_NAME = {stage["id"]: stage["name"] for stage in STAGES}
