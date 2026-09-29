"""
Tests for Quantum Recipe Generator (train.py & colab notebook).
"""
import json
from pathlib import Path
from quantum.recipe import RecipeGenerator
from quantum.constants import TRAIN_SCRIPT_FILE, COLAB_NOTEBOOK_FILE, RECIPE_FILE


def test_recipe_unsloth_generation(tmp_path):
    gen = RecipeGenerator(root_dir=tmp_path)
    recipe = gen.generate_recipe(
        framework="unsloth",
        quantization="4bit",
        lora_r=16,
        learning_rate=2e-4,
        epochs=3,
        batch_size=2,
        grad_accum=4,
        max_seq_length=1024,
    )

    assert recipe["framework"] == "unsloth"
    assert recipe["hyperparameters"]["lora_r"] == 16
    assert recipe["hyperparameters"]["lora_alpha"] == 32
    assert recipe["hyperparameters"]["max_seq_length"] == 1024

    # Check files created
    train_py = tmp_path / TRAIN_SCRIPT_FILE
    colab_nb = tmp_path / COLAB_NOTEBOOK_FILE
    recipe_json = tmp_path / RECIPE_FILE

    assert train_py.exists()
    assert colab_nb.exists()
    assert recipe_json.exists()

    # Check train.py content
    train_content = train_py.read_text(encoding="utf-8")
    assert "FastLanguageModel" in train_content
    assert "SFTTrainer" in train_content
    assert "1024" in train_content

    # Check colab notebook structure
    with open(colab_nb, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert "cells" in nb
    assert len(nb["cells"]) >= 4
    assert any("unsloth" in "".join(c["source"]) for c in nb["cells"])


def test_recipe_trl_generation(tmp_path):
    gen = RecipeGenerator(root_dir=tmp_path)
    recipe = gen.generate_recipe(framework="trl", quantization="4bit")
    assert recipe["framework"] == "trl"
    train_content = (tmp_path / TRAIN_SCRIPT_FILE).read_text(encoding="utf-8")
    assert "BitsAndBytesConfig" in train_content
    assert "LoraConfig" in train_content
