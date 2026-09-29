"""
Model Catalog and Recommendation Engine for Quantum.
Matches VRAM, context size, and task domain with optimal foundation models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantum.constants import MODEL_CHOICE_FILE, QUANTUM_DIR


SUPPORTED_MODELS: List[Dict[str, Any]] = [
    {
        "id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "huggingface_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "name": "Qwen 2.5 Coder 7B Instruct",
        "params": "7B",
        "family": "qwen2.5",
        "chat_template": "chatml",
        "min_vram_4bit_gb": 8.0,
        "min_vram_16bit_gb": 24.0,
        "best_for": ["code", "extraction", "json", "tool_calling", "general"],
        "description": "Excelente em JSON estrito, raciocínio lógico e extração estruturada de dados.",
    },
    {
        "id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "family": "llama3",
        "chat_template": "llama-3",
        "min_vram_4bit_gb": 9.0,
        "min_vram_16bit_gb": 24.0,
        "best_for": ["chat", "reasoning", "general", "agent"],
        "description": "Líder geral da categoria open-weight em conversação fluida e raciocínio.",
    },
    {
        "id": "unsloth/gemma-2-9b-it-bnb-4bit",
        "huggingface_id": "google/gemma-2-9b-it",
        "name": "Gemma 2 9B Instruct",
        "params": "9B",
        "family": "gemma2",
        "chat_template": "gemma",
        "min_vram_4bit_gb": 10.0,
        "min_vram_16bit_gb": 28.0,
        "best_for": ["reasoning", "summarization", "knowledge"],
        "description": "Altíssima densidade de conhecimento e qualidade de texto em português e inglês.",
    },
    {
        "id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "huggingface_id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen 2.5 3B Instruct",
        "params": "3B",
        "family": "qwen2.5",
        "chat_template": "chatml",
        "min_vram_4bit_gb": 4.5,
        "min_vram_16bit_gb": 12.0,
        "best_for": ["extraction", "json", "edge", "low_vram", "classification"],
        "description": "Leve e ultra-rápido, ideal para GPUs de entrada (RTX 3060/4060) ou ambientes edge.",
    },
    {
        "id": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "huggingface_id": "microsoft/Phi-3.5-mini-instruct",
        "name": "Phi 3.5 Mini Instruct (3.8B)",
        "params": "3.8B",
        "family": "phi3",
        "chat_template": "phi-3",
        "min_vram_4bit_gb": 5.0,
        "min_vram_16bit_gb": 14.0,
        "best_for": ["reasoning", "math", "code", "low_vram"],
        "description": "Modelo compacto treinado com dados sintéticos de alta qualidade e grande capacidade de raciocínio.",
    },
    {
        "id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "huggingface_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "params": "7B",
        "family": "qwen2.5",
        "chat_template": "chatml",
        "min_vram_4bit_gb": 8.5,
        "min_vram_16bit_gb": 24.0,
        "best_for": ["reasoning", "math", "analysis"],
        "description": "Especializado em Cadeia de Pensamento (CoT) e raciocínio analítico profundo.",
    },
    {
        "id": "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
        "huggingface_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2 1.7B Instruct",
        "params": "1.7B",
        "family": "llama",
        "chat_template": "chatml",
        "min_vram_4bit_gb": 3.0,
        "min_vram_16bit_gb": 8.0,
        "best_for": ["edge", "cpu", "low_vram", "classification"],
        "description": "Ideal para treinamento ultraleve, testes rápidos e CPU fine-tuning.",
    },
]


class ModelSelector:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)

    def recommend(
        self,
        available_vram_gb: float,
        task_type: str = "general",
        preferred_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommends best model given available VRAM and task.
        """
        # If user explicitly specified model id
        if preferred_model_id:
            for m in SUPPORTED_MODELS:
                if m["id"] == preferred_model_id or m["huggingface_id"] == preferred_model_id or m["family"] == preferred_model_id:
                    return {
                        "selected_model": m,
                        "selection_method": "manual",
                        "all_compatible": [m],
                    }

        # Filter by VRAM compatibility (at 4bit)
        # If available_vram_gb <= 0 (e.g. CPU), fallback to SmolLM2 or Qwen 3B
        effective_vram = available_vram_gb if available_vram_gb > 0 else 6.0

        compatible = [
            m for m in SUPPORTED_MODELS
            if m["min_vram_4bit_gb"] <= (effective_vram * 1.05)
        ]

        if not compatible:
            # Fallback to smallest
            compatible = [SUPPORTED_MODELS[-1]]

        # Score by task affinity
        def score_model(m: Dict[str, Any]) -> int:
            score = 0
            if task_type in m["best_for"]:
                score += 10
            # Preference for 7B/8B if VRAM allows
            if "7B" in m["params"] or "8B" in m["params"] or "9B" in m["params"]:
                score += 5
            return score

        compatible.sort(key=score_model, reverse=True)
        chosen = compatible[0]

        choice = {
            "selected_model": chosen,
            "selection_method": "auto_recommendation",
            "task_type": task_type,
            "effective_vram_gb": effective_vram,
            "alternatives": [m["name"] for m in compatible[1:4]],
            "chat_template": chosen["chat_template"],
            "base_model_name": chosen["name"],
            "huggingface_id": chosen["huggingface_id"],
            "unsloth_id": chosen["id"],
        }
        return choice

    def save_choice(self, choice_data: Dict[str, Any]) -> Path:
        out_path = self.root_dir / MODEL_CHOICE_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(choice_data, f, indent=2, ensure_ascii=False)
        return out_path
