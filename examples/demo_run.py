"""
Quantum Framework Demonstration Script.
Walks through all 11 stages of the Quantum LLM fine-tuning harness.
"""
import subprocess
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_command(cmd_args: list[str]):
    full_cmd = [sys.executable, "-m", "quantum.cli"] + cmd_args
    print(f"\n[EXEC] {' '.join(full_cmd)}")
    res = subprocess.run(full_cmd)
    if res.returncode != 0:
        print(f"[FAIL] Erro ao executar comando: {' '.join(cmd_args)}")
        sys.exit(res.returncode)


def main():
    print("================================================================")
    print("DEMO DO HARNESS QUANTUM - 11 ETAPAS DE FINE-TUNING")
    print("================================================================")

    # 0. Inicialização
    run_command(["init", "--name", "quantum-juridico", "--force"])
    run_command(["status"])

    # 1. Intenção
    run_command([
        "stage", "intent",
        "--task", "extraction",
        "--domain", "Direito Processual Civil",
        "--output-format", "json",
        "--description", "Extracao de metadados em peticoes iniciais"
    ])

    # 2. Ambiente
    run_command(["stage", "env", "--target", "colab", "--profile", "colab-free"])

    # 3. Dados & Guardrails (Privacidade On-Premises)
    run_command([
        "stage", "data-check",
        "--dataset", "examples/data/train_juridico.jsonl",
        "--schema", "alpaca"
    ])

    # 4. Escolha do Modelo
    run_command(["stage", "model-select"])

    # 5. Régua Baseline
    run_command([
        "stage", "baseline",
        "--eval-set", "examples/data/eval_juridico.jsonl",
        "--backend", "mock"
    ])

    # 6. Congelar Objetivo
    run_command([
        "stage", "freeze-objective",
        "--min-accuracy", "0.90",
        "--min-json-rate", "0.98"
    ])

    # 7. Receita
    run_command([
        "stage", "recipe",
        "--framework", "unsloth",
        "--quantization", "4bit",
        "--epochs", "3"
    ])

    # 8. Ensaio & Treino
    run_command(["stage", "dry-run", "--batches", "3"])
    run_command(["stage", "train", "--mock", "--steps", "30"])

    # 9. Avaliação
    run_command([
        "stage", "evaluate",
        "--mock-accuracy", "0.97",
        "--mock-json-rate", "1.0"
    ])

    # 10. Exportação
    run_command(["stage", "export", "--format", "gguf", "--quant", "q4_k_m"])

    # 11. Entrega
    run_command([
        "stage", "deliver",
        "--target", "ollama",
        "--tag", "quantum-juridico:latest"
    ])

    # Status Final
    run_command(["status"])

    print("\n================================================================")
    print("[SUCESSO] DEMONSTRACAO CONCLUIDA! TODOS OS ARTEFATOS FORAM GERADOS.")
    print("================================================================")


if __name__ == "__main__":
    main()
