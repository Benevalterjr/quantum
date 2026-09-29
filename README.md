# Quantum: Harness Framework de 11 Etapas para Fine-Tuning de LLMs

O **Quantum** é um framework no estilo *harness* (nos mesmos moldes do *It's Fine* e *Reversa*) projetado para conduzir engenheiros de IA e agentes de código (como Antigravity, Claude Code, Codex e Aider) através de um processo estruturado, seguro e rigoroso de 11 etapas para adaptação, fine-tuning e entrega de Modelos de Linguagem de Grande Porte (LLMs).

O objetivo primordial do Quantum é garantir **alta qualidade técnica**, **prevenção de OOM (Out Of Memory)** e **proteção absoluta de privacidade**: nenhum dado confidencial ou PII de treino é exposto ao contexto do agente de linguagem. Toda inspeção e cálculo de dados ocorrem localmente através de guardrails determinísticos que alimentam o harness apenas com estatísticas anônimas.

---

## As 11 Etapas do Quantum

| # | Etapa | Descrição | Artefato Gerado |
|---|---|---|---|
| **1** | **Intenção** | Entrevista técnica e definição de tarefa (classificação, extração, chat, raciocínio), domínio e formato de saída. | `.quantum/intent.json` |
| **2** | **Ambiente** | Detecção e perfil do hardware (VRAM, CUDA, CPU, RAM) ou nuvem (Colab, RunPod) com análise de headroom. | `.quantum/environment.json` |
| **3** | **Dados & Guardrails** | Validação de schema, encoding e tokenização local sem envio de PII à LLM. Apenas estatísticas anônimas. | `.quantum/telemetry.json` |
| **4** | **Escolha do Modelo** | Recomendação do modelo base ideal (Qwen, Llama, Gemma, Phi, etc.) de acordo com VRAM e objetivo. | `.quantum/model_choice.json` |
| **5** | **Régua (Baseline)** | Execução e registro de 10-20 casos de teste com o modelo base zero-shot para medir a pontuação inicial. | `.quantum/baseline_results.json` |
| **6** | **Congelar Objetivo** | Definição imutável dos critérios de aceitação e metas mínimas de qualidade e conformidade. | `.quantum/benchmark_spec.json` |
| **7** | **Receita** | Geração automática de scripts de treino autocontidos (LoRA/QLoRA) com Unsloth ou Hugging Face TRL. | `train.py`, `colab_train.ipynb`, `.quantum/recipe.json` |
| **8** | **Ensaio & Treino** | Ensaio (dry-run de 2-5 batches) para validação de VRAM e estabilidade; liberação e execução do treino. | `.quantum/dry_run.json`, `.quantum/training_log.json` |
| **9** | **Avaliação** | Reavaliação na mesma régua da Etapa 5 com comparação direta (*Before vs After*) e validação contra o objetivo congelado. | `.quantum/evaluation_report.json`, `evaluation_report.md` |
| **10** | **Exportação** | Merge dos pesos LoRA para Safetensors e quantização GGUF (Q4_K_M, Q8_0) pronta para inferência. | `.quantum/export_spec.json`, binários em `export/` |
| **11** | **Entrega** | Geração do `Modelfile` (Ollama), servidor vLLM / FastAPI e documentação técnica de consumo. | `Modelfile`, `serve_vllm.sh`, `HANDOVER.md` |

---

## Arquitetura de Privacidade dos Guardrails

```
[Arquivos Locais (.jsonl, .csv)]
             │
             ▼
   [quantum stage data-check]  <─── Executa 100% no terminal local (Offline)
             │
             ├──► Erros de schema, token length, contagem de amostras, p95/p99
             │
             ▼
   [Estatísticas Sanitizadas (JSON)] ───► Injetado no contexto do Agente (Sem PII)
```

---

## Uso Rápido (CLI)

```bash
# Inicializar um projeto Quantum
quantum init --name "suporte-juridico-llm"

# Verificar status da máquina de estados
quantum status

# Executar cada etapa sequencialmente
quantum stage intent --task extraction --domain "Direito Tributário" --output json
quantum stage env --target local
quantum stage data-check --dataset data/train.jsonl --schema alpaca
quantum stage model-select
quantum stage baseline --eval-set data/eval.jsonl
quantum stage freeze-objective --min-accuracy 0.95 --min-json-rate 0.99
quantum stage recipe --framework unsloth --quantization 4bit
quantum stage dry-run
quantum stage train
quantum stage evaluate
quantum stage export --format gguf --quant q4_k_m
quantum stage deliver --target ollama
```
