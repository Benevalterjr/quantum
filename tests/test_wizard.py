"""
Tests for Quantum Interactive Wizard.
"""
from pathlib import Path
from unittest.mock import patch
from quantum.wizard import QuantumWizard
from quantum.constants import INTENT_FILE, ENV_FILE, TELEMETRY_FILE, RECIPE_FILE, COLAB_NOTEBOOK_FILE


def test_wizard_express_colab(tmp_path):
    wiz = QuantumWizard(root_dir=tmp_path)

    # Mock user prompts for express mode
    with patch("rich.prompt.Prompt.ask") as mock_ask, \
         patch("rich.prompt.IntPrompt.ask") as mock_int_ask:
        mock_ask.side_effect = ["extraction", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"]
        mock_int_ask.side_effect = [3, 2048]

        wiz.run_express_colab()

    assert (tmp_path / COLAB_NOTEBOOK_FILE).exists()


def test_wizard_full_step_1_and_2(tmp_path):
    wiz = QuantumWizard(root_dir=tmp_path)

    # Test step 1
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.side_effect = ["test-wizard-proj", "1", "Direito Tributario", "json", "Descricao teste"]
        wiz.step_1_intent()

    assert (tmp_path / INTENT_FILE).exists()
    assert wiz.sm.is_stage_completed(1) is True

    # Test step 2
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.side_effect = ["1", "colab-free"]
        wiz.step_2_env()

    assert (tmp_path / ENV_FILE).exists()
    assert wiz.sm.is_stage_completed(2) is True
