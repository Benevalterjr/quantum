"""
Training Recipe Generator for Quantum (Stage 7).
Generates standalone train.py and colab_train.ipynb using Unsloth or TRL/PEFT.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from quantum.constants import (
    COLAB_NOTEBOOK_FILE,
    MODEL_CHOICE_FILE,
    RECIPE_FILE,
    TELEMETRY_FILE,
    TRAIN_SCRIPT_FILE,
    ENV_FILE,
)


class RecipeGenerator:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)

    def generate_recipe(
        self,
        framework: str = "unsloth",
        quantization: str = "4bit",
        lora_r: int = 16,
        lora_alpha: Optional[int] = None,
        learning_rate: float = 2e-4,
        epochs: int = 3,
        batch_size: Optional[int] = None,
        grad_accum: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculates hyperparams and generates train.py and notebook.
        """
        # Load environment specs
        env_file = self.root_dir / ENV_FILE
        env_data = {}
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
            except Exception:
                pass

        # Load telemetry
        telem_file = self.root_dir / TELEMETRY_FILE
        telem_data = {}
        if telem_file.exists():
            try:
                with open(telem_file, "r", encoding="utf-8") as f:
                    telem_data = json.load(f)
            except Exception:
                pass

        # Load model choice
        model_file = self.root_dir / MODEL_CHOICE_FILE
        model_data = {}
        if model_file.exists():
            try:
                with open(model_file, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
            except Exception:
                pass

        selected_model = model_data.get("selected_model", {})
        model_id = selected_model.get("id", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
        hf_model_id = selected_model.get("huggingface_id", "Qwen/Qwen2.5-Coder-7B-Instruct")
        chat_template = selected_model.get("chat_template", "chatml")

        final_seq_len = max_seq_length or telem_data.get("recommended_max_seq_length", 2048)
        final_bs = batch_size or env_data.get("recommended_batch_size", 2)
        final_ga = grad_accum or env_data.get("recommended_gradient_accumulation_steps", 4)
        alpha = lora_alpha or (lora_r * 2)

        recipe = {
            "stage": "recipe",
            "framework": framework,
            "quantization": quantization,
            "base_model": model_id,
            "huggingface_model": hf_model_id,
            "chat_template": chat_template,
            "hyperparameters": {
                "lora_r": lora_r,
                "lora_alpha": alpha,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "learning_rate": learning_rate,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": final_bs,
                "gradient_accumulation_steps": final_ga,
                "max_seq_length": final_seq_len,
                "warmup_ratio": 0.05,
                "weight_decay": 0.01,
                "optimizer": "paged_adamw_8bit" if quantization == "4bit" else "adamw_torch",
                "lr_scheduler_type": "cosine",
                "logging_steps": 1,
                "save_strategy": "epoch",
            },
            "output_dir": "checkpoints",
        }

        # 1. Generate train.py
        self._write_train_script(recipe)

        # 2. Generate colab_train.ipynb
        self._write_colab_notebook(recipe)

        # 3. Save recipe.json
        out_path = self.root_dir / RECIPE_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=2, ensure_ascii=False)

        return recipe

    def _write_train_script(self, recipe: Dict[str, Any]) -> None:
        hp = recipe["hyperparameters"]
        model_id = recipe["base_model"]
        framework = recipe["framework"]
        load_in_4bit = recipe["quantization"] == "4bit"

        script_content = f'''"""
Quantum Standalone Training Script.
Framework: {framework.upper()} | Model: {model_id}
Generated automatically by Quantum Harness.
"""
import os
import sys
import torch
from datasets import load_dataset
from transformers import TrainingArguments

# Configuration
MODEL_NAME = "{model_id}"
MAX_SEQ_LENGTH = {hp["max_seq_length"]}
LOAD_IN_4BIT = {load_in_4bit}
OUTPUT_DIR = "{recipe["output_dir"]}"

print("==================================================")
print("🚀 Quantum Training Runner")
print(f"📦 Model: {{MODEL_NAME}}")
print(f"⚡ Framework: {framework}")
print(f"📏 Max Seq Length: {{MAX_SEQ_LENGTH}}")
print("==================================================")

'''
        if framework == "unsloth":
            script_content += f'''try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
except ImportError:
    print("❌ Erro: unsloth não está instalado. Execute: pip install unsloth")
    sys.exit(1)

# 1. Carregar Modelo & Tokenizer com Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)

# 2. Configurar Adaptadores LoRA / QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r={hp["lora_r"]},
    target_modules={json.dumps(hp["target_modules"])},
    lora_alpha={hp["lora_alpha"]},
    lora_dropout={hp["lora_dropout"]},
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
'''
        else:
            script_content += f'''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

bnb_config = BitsAndBytesConfig(
    load_in_4bit={load_in_4bit},
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if LOAD_IN_4BIT else None,
    device_map="auto",
)

peft_config = LoraConfig(
    r={hp["lora_r"]},
    lora_alpha={hp["lora_alpha"]},
    lora_dropout={hp["lora_dropout"]},
    target_modules={json.dumps(hp["target_modules"])},
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
'''

        script_content += f'''
# 3. Carregar Dataset Local
data_file = sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl"
if not os.path.exists(data_file):
    print(f"❌ Arquivo de dataset não encontrado: {{data_file}}")
    sys.exit(1)

dataset = load_dataset("json", data_files=data_file, split="train")

# 4. Argumentos de Treinamento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size={hp["per_device_train_batch_size"]},
    gradient_accumulation_steps={hp["gradient_accumulation_steps"]},
    warmup_ratio={hp["warmup_ratio"]},
    num_train_epochs={hp["num_train_epochs"]},
    learning_rate={hp["learning_rate"]},
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps={hp["logging_steps"]},
    optim="{hp["optimizer"]}",
    weight_decay={hp["weight_decay"]},
    lr_scheduler_type="{hp["lr_scheduler_type"]}",
    seed=3407,
    save_strategy="epoch",
)

# 5. Inicializar SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text" if "text" in dataset.column_names else None,
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# 6. Executar Treinamento
print("🔥 Iniciando Treinamento LoRA...")
trainer_stats = trainer.train()

# 7. Salvar Adaptador Final
print(f"💾 Salvando adaptadores LoRA em {{OUTPUT_DIR}}/final_adapter...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
print("✅ Treinamento finalizado com sucesso!")
'''
        out_file = self.root_dir / TRAIN_SCRIPT_FILE
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(script_content)

    def _write_colab_notebook(self, recipe: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Creates interactive, turnkey Google Colab notebook (.ipynb)."""
        hp = recipe["hyperparameters"]
        model_id = recipe["base_model"]
        seq_len = hp["max_seq_length"]
        lora_r = hp["lora_r"]
        lora_alpha = hp["lora_alpha"]
        lr = hp["learning_rate"]
        epochs = hp["num_train_epochs"]
        bs = hp["per_device_train_batch_size"]
        ga = hp["gradient_accumulation_steps"]
        chat_template = recipe.get("chat_template", "chatml")

        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🌌 Quantum Harness: Fine-Tuning de LLM no Google Colab (Unsloth + LoRA)\n",
                    "\n",
                    "> **Gerado pelo framework Quantum** | Pronto para execução direta com 1 clique.\n",
                    "\n",
                    "### 📋 Especificações da Receita Gerada:\n",
                    f"- **Modelo Base:** `{model_id}`\n",
                    f"- **Comprimento Máximo de Sequência:** `{seq_len}` tokens\n",
                    f"- **Quantização:** `4-bit (NF4 BitsAndBytes)`\n",
                    f"- **LoRA:** `rank={lora_r}`, `alpha={lora_alpha}`, `dropout=0.05`\n",
                    f"- **Hiperparâmetros:** `epochs={epochs}`, `batch_size={bs}`, `grad_accum={ga}`, `lr={lr}`\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verificação do Ambiente de GPU\n",
                    "Certifique-se de que o Colab está conectado a uma GPU (Menu: *Ambiente de Execução > Alterar tipo de ambiente de execução > GPU T4, A100 ou L4*)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "import torch\n",
                    "print(f\"CUDA Disponível: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                    "    print(f\"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
                    "    print(f\"Suporte a BF16: {torch.cuda.is_bf16_supported()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Instalação das Dependências Otimizadas (Unsloth)\n",
                    "Instalação do Unsloth com kernels Triton compilados para velocidade 2-5x maior e 70% menos memória VRAM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%%capture\n",
                    "!pip install --no-deps \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                    "!pip install --no-deps trl peft accelerate bitsandbytes datasets sentencepiece tiktoken"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Carregamento do Modelo Base e Tokenizer\n",
                    f"Carrega `{model_id}` com quantização 4-bit para caber confortavelmente em GPUs de 15GB (T4 gratuita do Colab) com folga."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from unsloth import FastLanguageModel\n",
                    "import torch\n",
                    "\n",
                    f"max_seq_length = {seq_len}\n",
                    f"model_name = \"{model_id}\"\n",
                    "load_in_4bit = True\n",
                    "\n",
                    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                    "    model_name=model_name,\n",
                    "    max_seq_length=max_seq_length,\n",
                    "    load_in_4bit=load_in_4bit,\n",
                    ")\n",
                    "print(\"✔ Modelo base e tokenizer carregados com sucesso!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Configuração dos Adaptadores LoRA / QLoRA\n",
                    "Configuração dos módulos alvo de atenção e projeção MLP com *Gradient Checkpointing* ativado."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model = FastLanguageModel.get_peft_model(\n",
                    "    model,\n",
                    f"    r={lora_r},\n",
                    f"    target_modules={json.dumps(hp['target_modules'])},\n",
                    f"    lora_alpha={lora_alpha},\n",
                    f"    lora_dropout={hp['lora_dropout']},\n",
                    "    bias=\"none\",\n",
                    "    use_gradient_checkpointing=\"unsloth\",\n",
                    "    random_state=3407,\n",
                    ")\n",
                    "print(\"✔ Adaptadores LoRA configurados com sucesso!\")\n",
                    "model.print_trainable_parameters()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Carregamento do Dataset de Treino\n",
                    "Escolha a opção mais conveniente para carregar seu dataset:\n",
                    "- **Opção A:** Fazer upload do seu arquivo local `.jsonl` direto aqui\n",
                    "- **Opção B:** Montar o Google Drive\n",
                    "- **Opção C:** Criar um dataset de exemplo para teste rápido imediato"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from datasets import load_dataset\n",
                    "\n",
                    "# Descomente a opção desejada:\n",
                    "\n",
                    "# --- OPÇÃO A: Upload direto do seu arquivo .jsonl ---\n",
                    "# from google.colab import files\n",
                    "# uploaded = files.upload()\n",
                    "# dataset_path = list(uploaded.keys())[0]\n",
                    "\n",
                    "# --- OPÇÃO B: Carregar do Google Drive ---\n",
                    "# from google.colab import drive\n",
                    "# drive.mount('/content/drive')\n",
                    "# dataset_path = '/content/drive/MyDrive/meu_dataset.jsonl'\n",
                    "\n",
                    "# --- OPÇÃO C: Dataset embutido de teste imediato (Default) ---\n",
                    "dataset_path = 'train.jsonl'\n",
                    "if not os.path.exists(dataset_path):\n",
                    "    import json\n",
                    "    amostras = [\n",
                    "        {\"instruction\": \"Resuma o artigo legal.\", \"input\": \"Contrato de prestação de serviços nº 10.\", \"output\": \"{\"tipo\": \"serviço\", \"status\": \"ativo\"}\"},\n",
                    "        {\"instruction\": \"Identifique os valores do processo.\", \"input\": \"Ação de cobrança no valor de R$ 50.000,00.\", \"output\": \"{\"valor\": 50000.00, \"classe\": \"Cobrança\"}\"},\n",
                    "    ] * 20  # Replica para 40 amostras de teste\n",
                    "    with open(dataset_path, 'w', encoding='utf-8') as f:\n",
                    "        for a in amostras:\n",
                    "            f.write(json.dumps(a) + '\\n')\n",
                    "    print(f\"Dataset sintético criado em {dataset_path} para teste!\")\n",
                    "\n",
                    "dataset = load_dataset('json', data_files=dataset_path, split='train')\n",
                    "print(f\"✔ Dataset carregado com {len(dataset)} registros!\")\n",
                    "print(\"Amostra:\", dataset[0])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Formatação e Mapeamento dos Prompts\n",
                    "Formatação com template de instrução para evitar desperdício de tokens e garantir máscara correta."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "prompt_template = \"\"\"Abaixo está uma instrução que descreve uma tarefa, acompanhada por uma entrada que fornece mais contexto. Escreva uma resposta que complete apropriadamente o pedido.\n",
                    "\n",
                    "### Instrução:\n",
                    "{}\n",
                    "\n",
                    "### Entrada:\n",
                    "{}\n",
                    "\n",
                    "### Resposta:\n",
                    "{}\"\"\"\n",
                    "\n",
                    "EOS_TOKEN = tokenizer.eos_token\n",
                    "\n",
                    "def format_prompts(batch):\n",
                    "    instructions = batch.get(\"instruction\", [\"\"] * len(batch[\"output\"]))\n",
                    "    inputs = batch.get(\"input\", [\"\"] * len(batch[\"output\"]))\n",
                    "    outputs = batch.get(\"output\", [\"\"] * len(batch[\"output\"]))\n",
                    "    texts = []\n",
                    "    for instr, inp, out in zip(instructions, inputs, outputs):\n",
                    "        text = prompt_template.format(instr, inp, out) + EOS_TOKEN\n",
                    "        texts.append(text)\n",
                    "    return {\"text\": texts}\n",
                    "\n",
                    "formatted_dataset = dataset.map(format_prompts, batched=True)\n",
                    "print(\"✔ Dataset formatado com sucesso!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Treinamento com SFTTrainer\n",
                    "Execução do treinamento supervisionado com otimizador paged_adamw_8bit e logging contínuo."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from trl import SFTTrainer\n",
                    "from transformers import TrainingArguments\n",
                    "\n",
                    "# Medição inicial de VRAM\n",
                    "gpu_stats = torch.cuda.get_device_properties(0)\n",
                    "start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n",
                    "print(f\"VRAM Reservada Inicial: {start_gpu_memory} GB de {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)} GB\")\n",
                    "\n",
                    "training_args = TrainingArguments(\n",
                    f"    per_device_train_batch_size={bs},\n",
                    f"    gradient_accumulation_steps={ga},\n",
                    f"    warmup_ratio={hp['warmup_ratio']},\n",
                    f"    num_train_epochs={epochs},\n",
                    f"    learning_rate={lr},\n",
                    "    fp16=not torch.cuda.is_bf16_supported(),\n",
                    "    bf16=torch.cuda.is_bf16_supported(),\n",
                    f"    logging_steps={hp['logging_steps']},\n",
                    f"    optim=\"{hp['optimizer']}\",\n",
                    f"    weight_decay={hp['weight_decay']},\n",
                    f"    lr_scheduler_type=\"{hp['lr_scheduler_type']}\",\n",
                    "    seed=3407,\n",
                    "    output_dir=\"checkpoints\",\n",
                    "    report_to=\"none\",\n",
                    ")\n",
                    "\n",
                    "trainer = SFTTrainer(\n",
                    "    model=model,\n",
                    "    tokenizer=tokenizer,\n",
                    "    train_dataset=formatted_dataset,\n",
                    "    dataset_text_field=\"text\",\n",
                    "    max_seq_length=max_seq_length,\n",
                    "    dataset_num_proc=2,\n",
                    "    packing=False,\n",
                    "    args=training_args,\n",
                    ")\n",
                    "\n",
                    "print(\"🔥 Iniciando treinamento do modelo...\")\n",
                    "trainer_stats = trainer.train()\n",
                    "\n",
                    "# Medição final de VRAM e tempo\n",
                    "used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n",
                    "print(f\"✔ Treinamento concluído em {trainer_stats.metrics['train_runtime']:.1f} segundos!\")\n",
                    "print(f\"Pico de VRAM utilizada: {used_memory} GB\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Teste Rápido de Inferência (Zero Latência no Colab)\n",
                    "Testa o modelo adaptado diretamente na memória da GPU para validar a qualidade da resposta."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "FastLanguageModel.for_inference(model) # 2x mais rápido na inferência\n",
                    "\n",
                    "inputs = tokenizer(\n",
                    "    [\n",
                    "        prompt_template.format(\n",
                    "            \"Identifique os valores do processo.\", # Instrução de teste\n",
                    "            \"Processo nº 4002/2026 com execução de título no valor de R$ 98.400,00 em Curitiba.\", # Entrada\n",
                    "            \"\", # Resposta vazia para a LLM completar\n",
                    "        )\n",
                    "    ],\n",
                    "    return_tensors=\"pt\",\n",
                    ").to(\"cuda\")\n",
                    "\n",
                    "outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)\n",
                    "res = tokenizer.batch_decode(outputs)\n",
                    "print(\"--- RESPOSTA GERADA PELO MODELO TREINADO ---\")\n",
                    "print(res[0].split(\"### Resposta:\")[-1].replace(EOS_TOKEN, \"\").strip())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Exportação Automática para Ollama (GGUF Quantizado)\n",
                    "O Unsloth permite exportar o modelo treinado diretamente para o formato **GGUF (Q4_K_M)** em 1 passo, pronto para rodar localmente no Ollama ou vLLM!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Salvar adaptadores LoRA locais primeiro\n",
                    "model.save_pretrained(\"lora_adapter\")\n",
                    "tokenizer.save_pretrained(\"lora_adapter\")\n",
                    "\n",
                    "# Exportar direto para binário GGUF (q4_k_m)\n",
                    "print(\"📦 Exportando para GGUF quantizado (q4_k_m)... isso levará de 1 a 2 minutos...\")\n",
                    "model.save_pretrained_gguf(\"quantum_model\", tokenizer, quantization_method=\"q4_k_m\")\n",
                    "print(\"✔ Exportação concluída com sucesso!\")\n",
                    "!ls -lh quantum_model*"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 10. Salvar no Google Drive ou Fazer Download\n",
                    "Mova o binário gerado para o seu Google Drive para não perder o modelo quando o Colab for desconectado."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Salvar no Google Drive:\n",
                    "# from google.colab import drive\n",
                    "# drive.mount('/content/drive')\n",
                    "# !cp quantum_model* /content/drive/MyDrive/\n",
                    "# print(\"✔ Modelo salvo no seu Google Drive!\")\n",
                    "\n",
                    "# Ou download direto para sua máquina:\n",
                    "# from google.colab import files\n",
                    "# files.download('quantum_model-unsloth.Q4_K_M.gguf')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 11. Modelfile para Execução no Ollama Local\n",
                    "Copie o conteúdo abaixo para um arquivo chamado `Modelfile` no seu computador e execute `ollama create quantum-model -f Modelfile`:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "modelfile_content = \"\"\"FROM ./quantum_model-unsloth.Q4_K_M.gguf\n",
                    "PARAMETER temperature 0.2\n",
                    "PARAMETER stop \"<|im_end|>\"\n",
                    "PARAMETER stop \"<|endoftext|>\"\n",
                    "\"\"\"\n",
                    "print(modelfile_content)\n",
                    "with open(\"Modelfile\", \"w\") as f:\n",
                    "    f.write(modelfile_content)"
                ]
            }
        ]

        notebook_data = {
            "cells": cells,
            "metadata": {
                "accelerator": "GPU",
                "colab": {
                    "gpuType": "T4",
                    "provenance": []
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        nb_file = output_path or (self.root_dir / COLAB_NOTEBOOK_FILE)
        with open(nb_file, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=2, ensure_ascii=False)
        return nb_file
