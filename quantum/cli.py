"""
Quantum CLI: Privacy-First 11-Stage Fine-Tuning Harness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quantum.constants import (
    INTENT_FILE,
    QUANTUM_DIR,
    STAGES,
)
from quantum.state_machine import QuantumStateMachine, StageValidationError
from quantum.hardware import HardwareInspector
from quantum.guardrails.anonymizer import DataCheckGuardrail
from quantum.models import ModelSelector
from quantum.baseline import BaselineEvaluator
from quantum.benchmark import ObjectiveFreezer
from quantum.recipe import RecipeGenerator
from quantum.dry_run import DryRunTester
from quantum.trainer import TrainingOrchestrator
from quantum.evaluation import Evaluator
from quantum.exporter import ModelExporter
from quantum.delivery import Deliverer


app = typer.Typer(
    name="quantum",
    help="Quantum: Harness Framework de 11 Etapas para Fine-Tuning de LLMs com Privacidade Rigorosa.",
    no_args_is_help=True,
)
stage_app = typer.Typer(name="stage", help="Executa etapas individuais do harness Quantum.")
workflow_app = typer.Typer(name="workflow", help="Gerenciamento do ciclo de vida e rollback do harness.")

app.add_typer(stage_app, name="stage")
app.add_typer(workflow_app, name="workflow")

console = Console(legacy_windows=False)


@app.command()
def init(
    name: str = typer.Option("quantum-project", "--name", "-n", help="Nome do projeto de fine-tuning"),
    force: bool = typer.Option(False, "--force", "-f", help="Forçar reinicialização do estado"),
):
    """Inicializa o projeto Quantum e a máquina de estados das 11 etapas."""
    sm = QuantumStateMachine()
    sm.init_project(project_name=name, force=force)
    rprint(Panel.fit(
        f"[bold cyan]Quantum Harness Inicializado com Sucesso![/bold cyan]\n"
        f"[green]Projeto:[/green] {name}\n"
        f"[yellow]Diretorio de Estado:[/yellow] .quantum/\n"
        f"[white]Proximo passo:[/white] [bold]quantum stage intent[/bold] para iniciar a Etapa 1.",
        title="Quantum Protocol",
        border_style="cyan"
    ))


@app.command()
def status():
    """Exibe o painel visual das 11 etapas, estado atual e artefatos."""
    sm = QuantumStateMachine()
    summary = sm.get_summary_table()
    current_stage = sm.current_stage_id

    table = Table(title=f"Status do Harness Quantum: [bold green]{sm.project_name}[/bold green]", border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Etapa", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Artefato", style="dim")
    table.add_column("No Disco?", justify="center")

    for row in summary:
        s_id = row["id"]
        status = row["status"]
        if status == "completed":
            status_text = "[bold green][OK] Concluido[/bold green]"
        elif s_id == current_stage:
            status_text = "[bold yellow]-> Atual[/bold yellow]"
        else:
            status_text = "[dim]Pendente[/dim]"

        disk_icon = "[green][OK][/green]" if row["artifact_exists"] else "[dim]-[/dim]"
        table.add_row(str(s_id), row["title"], status_text, row["artifact"], disk_icon)

    console.print(table)


# ==============================================================================
# ETAPA 1: INTENÇÃO
# ==============================================================================
@stage_app.command("intent")
def stage_intent(
    task: str = typer.Option(..., "--task", "-t", help="Tipo de tarefa (ex: extraction, classification, reasoning, chat)"),
    domain: str = typer.Option(..., "--domain", "-d", help="Domínio de aplicação (ex: Jurídico, Finanças, Medicina)"),
    output_format: str = typer.Option("json", "--output-format", "-o", help="Formato esperado de saída (json, markdown, text)"),
    description: str = typer.Option("", "--description", help="Detalhes e regras de negócio da tarefa"),
):
    """Etapa 1: Entrevista técnica e registro da intenção do projeto."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(1)
    if not allowed:
        rprint(f"[bold red]Erro:[/bold red] {reason}")
        raise typer.Exit(1)

    intent_data = {
        "task": task,
        "domain": domain,
        "output_format": output_format,
        "description": description,
    }

    intent_file = Path(INTENT_FILE)
    intent_file.parent.mkdir(parents=True, exist_ok=True)
    with open(intent_file, "w", encoding="utf-8") as f:
        json.dump(intent_data, f, indent=2, ensure_ascii=False)

    sm.complete_stage(1, metadata=intent_data)
    rprint(f"[bold green][OK] Etapa 1 (Intencao) concluida![/bold green] Registrado em {INTENT_FILE}")
    rprint("[dim]Próxima etapa: [bold]quantum stage env[/bold][/dim]")


# ==============================================================================
# ETAPA 2: AMBIENTE
# ==============================================================================
@stage_app.command("env")
def stage_env(
    target: str = typer.Option("local", "--target", "-t", help="Alvo de treinamento: local, colab, runpod, cloud"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil de nuvem (ex: colab-free, colab-pro, runpod-4090)"),
    vram_gb: Optional[float] = typer.Option(None, "--vram-gb", help="Override manual de VRAM em GB"),
):
    """Etapa 2: Inspeção de hardware (VRAM, CUDA, CPU, RAM) ou perfil de nuvem."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(2)
    if not allowed:
        rprint(f"[bold red]Erro de Portão:[/bold red] {reason}")
        raise typer.Exit(1)

    inspector = HardwareInspector()
    env_spec = inspector.profile_environment(target=target, cloud_profile=profile, custom_vram_gb=vram_gb)
    saved_path = inspector.save_environment(env_spec)

    sm.complete_stage(2, metadata={"target": target, "gpu": env_spec["gpu_name"], "vram_gb": env_spec["vram_gb"]})
    rprint(f"[bold green][OK] Etapa 2 (Ambiente) concluida![/bold green] Salvo em {saved_path}")
    rprint(f"  [cyan]Hardware:[/cyan] {env_spec['gpu_name']} ({env_spec['vram_gb']} GB VRAM)")
    rprint(f"  [cyan]Quantizacao recomendada:[/cyan] {env_spec['recommended_quantization'].upper()}")
    rprint("[dim]Proxima etapa: [bold]quantum stage data-check[/bold][/dim]")


# ==============================================================================
# ETAPA 3: DADOS & GUARDRAILS (PRIVACIDADE ESTRITA)
# ==============================================================================
@stage_app.command("data-check")
def stage_data_check(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Caminho para o dataset (.jsonl, .json ou .csv)"),
    val_dataset: Optional[str] = typer.Option(None, "--val-dataset", help="Caminho opcional para dataset de validacao"),
    schema: str = typer.Option("auto", "--schema", "-s", help="Schema esperado: auto, alpaca, chatml, sharegpt"),
):
    """Etapa 3: Guardrail de dados 100% offline. Zero vazamento de texto bruto ou PII."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(3)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    guardrail = DataCheckGuardrail()
    try:
        telemetry = guardrail.inspect_dataset(dataset, val_dataset_path=val_dataset, schema=schema)
    except Exception as e:
        rprint(f"[bold red]Erro no guardrail de dados:[/bold red] {e}")
        raise typer.Exit(1)

    ds_info = telemetry["dataset_info"]
    tok_info = telemetry["token_distribution"]

    if not ds_info["is_valid"]:
        rprint(f"[bold red][FAIL] Validacao de dados falhou com {ds_info['invalid_records']} erros de schema![/bold red]")
        for err in ds_info["errors"][:5]:
            rprint(f"  [yellow]* {err}[/yellow]")
        raise typer.Exit(1)

    sm.complete_stage(3, metadata={
        "total_records": ds_info["total_records"],
        "schema": ds_info["detected_schema"],
        "recommended_seq_len": telemetry["recommended_max_seq_length"],
    })

    rprint(f"[bold green][OK] Etapa 3 (Dados & Guardrails) concluida com sucesso![/bold green]")
    rprint(f"  [cyan]Amostras validas:[/cyan] {ds_info['valid_records']} / {ds_info['total_records']}")
    rprint(f"  [cyan]Schema detectado:[/cyan] {ds_info['detected_schema'].upper()}")
    rprint(f"  [cyan]Distribuicao de Tokens:[/cyan] Min={tok_info['min']} | Media={tok_info['mean']} | P95={tok_info['p95']} | P99={tok_info['p99']}")
    rprint(f"  [green]Garantia de Privacidade:[/green] {telemetry['privacy_guarantee']}")
    rprint("[dim]Proxima etapa: [bold]quantum stage model-select[/bold][/dim]")


# ==============================================================================
# ETAPA 4: ESCOLHA DO MODELO
# ==============================================================================
@stage_app.command("model-select")
def stage_model_select(
    model_id: Optional[str] = typer.Option(None, "--model-id", "-m", help="Especificar modelo manualmente"),
):
    """Etapa 4: Recomenda e seleciona o foundation model ideal."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(4)
    if not allowed:
        rprint(f"[bold red]Erro de Portão:[/bold red] {reason}")
        raise typer.Exit(1)

    # Read env and intent
    env_info = sm.get_stage_info("env")["metadata"]
    intent_info = sm.get_stage_info("intent")["metadata"]
    vram = env_info.get("vram_gb", 16.0)
    task = intent_info.get("task", "general")

    selector = ModelSelector()
    choice = selector.recommend(available_vram_gb=vram, task_type=task, preferred_model_id=model_id)
    selector.save_choice(choice)

    sm.complete_stage(4, metadata={
        "model_name": choice["base_model_name"],
        "unsloth_id": choice["unsloth_id"],
        "chat_template": choice["chat_template"],
    })

    rprint(f"[bold green][OK] Etapa 4 (Escolha do Modelo) concluida![/bold green]")
    rprint(f"  [cyan]Modelo Selecionado:[/cyan] {choice['base_model_name']}")
    rprint(f"  [cyan]Template:[/cyan] {choice['chat_template']}")
    rprint("[dim]Proxima etapa: [bold]quantum stage baseline[/bold][/dim]")


# ==============================================================================
# ETAPA 5: RÉGUA BASELINE
# ==============================================================================
@stage_app.command("baseline")
def stage_baseline(
    eval_set: str = typer.Option(..., "--eval-set", "-e", help="Dataset com 10-20 amostras de teste"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Backend de execucao: mock, ollama, api"),
):
    """Etapa 5: Executa a regua de avaliacao zero-shot no modelo base."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(5)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    model_name = sm.get_stage_info("model-select")["metadata"].get("model_name", "Base Model")
    evaluator = BaselineEvaluator()
    results = evaluator.run_baseline(eval_set, backend=backend, base_model_name=model_name)

    sm.complete_stage(5, metadata=results["metrics"])
    rprint(f"[bold green][OK] Etapa 5 (Regua Baseline) concluida![/bold green]")
    rprint(f"  [cyan]Acuracia Zero-Shot Inicial:[/cyan] {results['metrics']['accuracy']*100:.1f}%")
    rprint(f"  [cyan]Taxa de JSON Valido Inicial:[/cyan] {results['metrics']['valid_json_rate']*100:.1f}%")
    rprint("[dim]Proxima etapa: [bold]quantum stage freeze-objective[/bold][/dim]")


# ==============================================================================
# ETAPA 6: CONGELAR OBJETIVO
# ==============================================================================
@stage_app.command("freeze-objective")
def stage_freeze_objective(
    min_accuracy: float = typer.Option(0.90, "--min-accuracy", "-a", help="Acuracia minima requerida (0.0 a 1.0)"),
    min_json_rate: float = typer.Option(0.98, "--min-json-rate", "-j", help="Taxa minima de JSON valido (0.0 a 1.0)"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Notas ou justificativa das metas"),
):
    """Etapa 6: Trava os criterios de aceitacao imutaveis para producao."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(6)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    freezer = ObjectiveFreezer()
    spec = freezer.freeze_objective(min_accuracy=min_accuracy, min_json_rate=min_json_rate, notes=notes)

    sm.complete_stage(6, metadata={
        "min_accuracy": min_accuracy,
        "min_json_rate": min_json_rate,
        "signature": spec["spec_signature"][:12],
    })

    rprint(f"[bold green][OK] Etapa 6 (Congelar Objetivo) concluida e travada com hash seguro![/bold green]")
    rprint(f"  [cyan]Meta Acuracia:[/cyan] >= {min_accuracy*100:.1f}%")
    rprint(f"  [cyan]Meta JSON Valido:[/cyan] >= {min_json_rate*100:.1f}%")
    rprint(f"  [dim]Assinatura:[/dim] {spec['spec_signature']}")
    rprint("[dim]Proxima etapa: [bold]quantum stage recipe[/bold][/dim]")


# ==============================================================================
# ETAPA 7: RECEITA
# ==============================================================================
@stage_app.command("recipe")
def stage_recipe(
    framework: str = typer.Option("unsloth", "--framework", "-f", help="Framework: unsloth ou trl"),
    quantization: str = typer.Option("4bit", "--quantization", "-q", help="Quantizacao: 4bit, 8bit, none"),
    epochs: int = typer.Option(3, "--epochs", help="Numero de epocas de treinamento"),
    lora_r: int = typer.Option(16, "--lora-r", help="Rank LoRA"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Taxa de aprendizado"),
):
    """Etapa 7: Gera os scripts independentes train.py e colab_train.ipynb."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(7)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    generator = RecipeGenerator()
    recipe = generator.generate_recipe(
        framework=framework,
        quantization=quantization,
        epochs=epochs,
        lora_r=lora_r,
        learning_rate=learning_rate,
    )

    sm.complete_stage(7, metadata={
        "framework": framework,
        "quantization": quantization,
        "epochs": epochs,
    })

    rprint(f"[bold green][OK] Etapa 7 (Receita) gerada com sucesso![/bold green]")
    rprint(f"  [cyan]Script de Treino:[/cyan] train.py")
    rprint(f"  [cyan]Notebook Colab:[/cyan] colab_train.ipynb")
    rprint("[dim]Proximos passos: [bold]quantum stage dry-run[/bold] seguido de [bold]quantum stage train[/bold][/dim]")


# ==============================================================================
# ETAPA 8: ENSAIO & TREINO
# ==============================================================================
@stage_app.command("dry-run")
def stage_dry_run(
    batches: int = typer.Option(3, "--batches", "-b", help="Numero de batches para o ensaio de VRAM"),
    force: bool = typer.Option(False, "--force", help="Forcar aprovacao mesmo se houver risco"),
):
    """Etapa 8.1 (Ensaio): Testa 2 a 5 batches para prevenir estouro de VRAM (CUDA OOM)."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(8)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    tester = DryRunTester()
    res = tester.run_dry_run(batches=batches, force_pass=force)

    if res["status"] != "PASSED":
        rprint(f"[bold red][FAIL] Ensaio Reprovado:[/bold red] {res['verdict']}")
        raise typer.Exit(1)

    rprint(f"[bold green][OK] Ensaio de VRAM Aprovado com Sucesso![/bold green]")
    rprint(f"  [cyan]Pico estimado:[/cyan] {res['estimated_peak_vram_gb']} GB / {res['vram_total_gb']} GB")
    rprint(f"  [cyan]Folga (Headroom):[/cyan] {res['headroom_gb']} GB")
    rprint("[dim]Treinamento liberado. Execute: [bold]quantum stage train[/bold][/dim]")


@stage_app.command("train")
def stage_train(
    mock: bool = typer.Option(True, "--mock/--live", help="Executar modo simulado para teste de harness ou treino real"),
    steps: int = typer.Option(50, "--steps", help="Número de steps de treinamento"),
):
    """Etapa 8.2 (Treinamento): Orquestra o treinamento completo dos adaptadores LoRA."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(8)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    orchestrator = TrainingOrchestrator()
    try:
        record = orchestrator.execute_training(mock=mock, steps=steps, verify_dry_run=True)
    except Exception as e:
        rprint(f"[bold red]Falha no treinamento:[/bold red] {e}")
        raise typer.Exit(1)

    sm.complete_stage(8, metadata={
        "initial_loss": record["initial_loss"],
        "final_loss": record["final_loss"],
        "total_steps": record["total_steps"],
    })

    rprint(f"[bold green][OK] Etapa 8 (Ensaio & Treino) concluida com exito![/bold green]")
    rprint(f"  [cyan]Perda (Loss):[/cyan] {record['initial_loss']} -> {record['final_loss']} (-{record['loss_reduction_pct']}%)")
    rprint(f"  [cyan]Adaptador salvo em:[/cyan] {record['adapter_path']}")
    rprint("[dim]Proxima etapa: [bold]quantum stage evaluate[/bold][/dim]")


# ==============================================================================
# ETAPA 9: AVALIAÇÃO
# ==============================================================================
@stage_app.command("evaluate")
def stage_evaluate(
    accuracy: float = typer.Option(0.96, "--mock-accuracy", help="Acuracia simulada pos-treino para testes"),
    json_rate: float = typer.Option(1.0, "--mock-json-rate", help="Taxa simulada de JSON valido para testes"),
):
    """Etapa 9: Compara o modelo treinado com a regua inicial e valida contra as metas."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(9)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    evaluator = Evaluator()
    report = evaluator.evaluate_model(mock_accuracy=accuracy, mock_valid_json_rate=json_rate)

    sm.complete_stage(9, metadata={
        "status": report["overall_status"],
        "accuracy": report["metrics"]["fine_tuned"]["accuracy"],
        "accuracy_gain": report["metrics"]["delta"]["accuracy_gain"],
    })

    status_color = "green" if report["acceptance_criteria_met"] else "red"
    rprint(f"[bold {status_color}][OK] Etapa 9 (Avaliacao) concluida![/bold {status_color}]")
    rprint(f"  [cyan]Veredito:[/cyan] [{status_color}]{report['overall_status']}[/{status_color}]")
    rprint(f"  [cyan]Ganho de Acuracia:[/cyan] +{report['metrics']['delta']['accuracy_gain']*100:.1f}%")
    rprint(f"  [cyan]Relatorio gerado:[/cyan] evaluation_report.md")
    rprint("[dim]Proxima etapa: [bold]quantum stage export[/bold][/dim]")


# ==============================================================================
# ETAPA 10: EXPORTAÇÃO
# ==============================================================================
@stage_app.command("export")
def stage_export(
    export_format: str = typer.Option("gguf", "--format", "-f", help="Formato: gguf ou safetensors"),
    quant: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantizacao GGUF: q4_k_m, q8_0, fp16"),
    name: str = typer.Option("quantum_model", "--name", "-n", help="Nome do arquivo de saida"),
):
    """Etapa 10: Faz o merge dos pesos LoRA e exporta para GGUF ou Safetensors."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(10)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    exporter = ModelExporter()
    res = exporter.export_model(export_format=export_format, quantization_type=quant, output_name=name)

    sm.complete_stage(10, metadata={
        "artifact": res["artifact_path"],
        "format": export_format,
        "quantization": quant,
    })

    rprint(f"[bold green][OK] Etapa 10 (Exportacao) concluida com sucesso![/bold green]")
    rprint(f"  [cyan]Artefato:[/cyan] {res['artifact_path']} (~{res['size_gb']} GB)")
    rprint("[dim]Proxima etapa final: [bold]quantum stage deliver[/bold][/dim]")


# ==============================================================================
# ETAPA 11: ENTREGA
# ==============================================================================
@stage_app.command("deliver")
def stage_deliver(
    target: str = typer.Option("ollama", "--target", "-t", help="Target principal de entrega: ollama, vllm, fastapi"),
    model_tag: str = typer.Option("quantum-model:latest", "--tag", help="Tag do modelo para deploy"),
):
    """Etapa 11: Gera Modelfile para Ollama, script de servico vLLM e HANDOVER.md."""
    sm = QuantumStateMachine()
    allowed, reason = sm.can_enter_stage(11)
    if not allowed:
        rprint(f"[bold red]Erro de Portao:[/bold red] {reason}")
        raise typer.Exit(1)

    deliverer = Deliverer()
    res = deliverer.prepare_delivery(service_target=target, model_tag=model_tag)

    sm.complete_stage(11, metadata={"status": res["status"], "tag": model_tag})

    rprint(Panel.fit(
        f"[bold green]PROJETO QUANTUM CONCLUIDO E ENTREGUE COM SUCESSO![/bold green]\n\n"
        f"[cyan]Artefatos Prontos:[/cyan]\n"
        f"  * Modelfile (Ollama): [bold]Modelfile[/bold]\n"
        f"  * Launcher vLLM: [bold]serve_vllm.sh[/bold]\n"
        f"  * API FastAPI: [bold]serve_api.py[/bold]\n"
        f"  * Documentacao Tecnica: [bold]HANDOVER.md[/bold]\n\n"
        f"[yellow]Para carregar no Ollama agora:[/yellow]\n"
        f"  ollama create quantum-model -f Modelfile\n"
        f"  ollama run quantum-model",
        title="11 Etapas Finalizadas",
        border_style="green"
    ))


# ==============================================================================
# WORKFLOW COMMANDS
# ==============================================================================
@workflow_app.command("rollback")
def workflow_rollback(
    to: int = typer.Option(..., "--to", "-t", help="Numero da etapa para rollback (1 a 11)"),
):
    """Executa rollback para uma etapa anterior, invalidando as etapas subsequentes."""
    sm = QuantumStateMachine()
    sm.rollback_to(to)
    rprint(f"[bold yellow]Rollback realizado com sucesso para a Etapa {to}![/bold yellow]")
    rprint("[dim]Use [bold]quantum status[/bold] para conferir o novo estado.[/dim]")


@workflow_app.command("reset")
def workflow_reset():
    """Reseta todo o estado do projeto para a Etapa 1."""
    sm = QuantumStateMachine()
    sm.init_project(force=True)
    rprint("[bold red]Estado do Quantum resetado para o inicio![/bold red]")


if __name__ == "__main__":
    app()
