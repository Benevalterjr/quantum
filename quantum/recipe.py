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

    def _write_colab_notebook(self, recipe: Dict[str, Any]) -> None:
        """Creates interactive Google Colab notebook (.ipynb)."""
        hp = recipe["hyperparameters"]
        model_id = recipe["base_model"]
        seq_len = hp["max_seq_length"]

        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🌌 Quantum: Fine-Tuning de LLM com Unsloth (LoRA/QLoRA)\n",
                    "Notebook gerado automaticamente pelo **Quantum Harness**.\n",
                    "- **Modelo:** `" + model_id + "`\n",
                    "- **Max Seq Length:** `" + str(seq_len) + "`\n",
                    "- **Quantização:** `" + recipe["quantization"] + "`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Instalar dependências otimizadas (Unsloth + PyTorch)\n",
                    "!pip install --no-deps \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                    "!pip install --no-deps trl peft accelerate bitsandbytes datasets"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Carregar modelo base e tokenizer\n",
                    "from unsloth import FastLanguageModel\n",
                    "import torch\n\n",
                    f'max_seq_length = {seq_len}\n',
                    f'model_name = "{model_id}"\n',
                    "load_in_4bit = True\n\n",
                    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                    "    model_name=model_name,\n",
                    "    max_seq_length=max_seq_length,\n",
                    "    load_in_4bit=load_in_4bit,\n",
                    ")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Configurar adaptadores LoRA\n",
                    "model = FastLanguageModel.get_peft_model(\n",
                    "    model,\n",
                    f"    r={hp['lora_r']},\n",
                    f"    target_modules={json.dumps(hp['target_modules'])},\n",
                    f"    lora_alpha={hp['lora_alpha']},\n",
                    f"    lora_dropout={hp['lora_dropout']},\n",
                    "    bias='none',\n",
                    "    use_gradient_checkpointing='unsloth',\n",
                    "    random_state=3407,\n",
                    ")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Carregar dataset e treinar\n",
                    "from datasets import load_dataset\n",
                    "from trl import SFTTrainer\n",
                    "from transformers import TrainingArguments\n\n",
                    "# Substitua pelo caminho do seu dataset enviado para o Colab\n",
                    "dataset = load_dataset('json', data_files='train.jsonl', split='train')\n\n",
                    "trainer = SFTTrainer(\n",
                    "    model=model,\n",
                    "    tokenizer=tokenizer,\n",
                    "    train_dataset=dataset,\n",
                    "    max_seq_length=max_seq_length,\n",
                    "    args=TrainingArguments(\n",
                    f"        per_device_train_batch_size={hp['per_device_train_batch_size']},\n",
                    f"        gradient_accumulation_steps={hp['gradient_accumulation_steps']},\n",
                    f"        warmup_ratio={hp['warmup_ratio']},\n",
                    f"        num_train_epochs={hp['num_train_epochs']},\n",
                    f"        learning_rate={hp['learning_rate']},\n",
                    "        fp16=not torch.cuda.is_bf16_supported(),\n",
                    "        bf16=torch.cuda.is_bf16_supported(),\n",
                    "        logging_steps=1,\n",
                    f"        optim='{hp['optimizer']}',\n",
                    f"        lr_scheduler_type='{hp['lr_scheduler_type']}',\n",
                    "        seed=3407,\n",
                    "        output_dir='checkpoints',\n",
                    "    ),\n",
                    ")\n",
                    "trainer.train()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Exportar para GGUF (Ollama)\n",
                    "# unsloth permite salvar diretamente em GGUF 4bit\n",
                    "model.save_pretrained_gguf('export/model', tokenizer, quantization_method='q4_k_m')\n",
                    "print('Exportado com sucesso para export/model!')"
                ]
            }
        ]

        notebook_data = {
            "cells": cells,
            "metadata": {
                "language_info": {"name": "python"},
                "accelerator": "GPU"
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        nb_file = self.root_dir / COLAB_NOTEBOOK_FILE
        with open(nb_file, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=2, ensure_ascii=False)
