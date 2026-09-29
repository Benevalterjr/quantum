"""
Tests for Quantum State Machine and Harness Transition Gates.
"""
import json
import pytest
from pathlib import Path

from quantum.state_machine import QuantumStateMachine, StageValidationError
from quantum.constants import INTENT_FILE, ENV_FILE, TELEMETRY_FILE


def test_initial_state(tmp_path):
    sm = QuantumStateMachine(root_dir=tmp_path)
    sm.init_project("test-project")
    assert sm.current_stage_id == 1
    assert sm.project_name == "test-project"
    assert sm.is_stage_completed(1) is False


def test_cannot_skip_stages(tmp_path):
    sm = QuantumStateMachine(root_dir=tmp_path)
    sm.init_project("test-project")

    # Stage 1 is allowed
    allowed, _ = sm.can_enter_stage(1)
    assert allowed is True

    # Stage 2 is blocked because Stage 1 is not completed
    allowed, reason = sm.can_enter_stage(2)
    assert allowed is False
    assert "requer que a etapa 1" in reason

    # Attempting to complete Stage 3 directly must raise StageValidationError
    with pytest.raises(StageValidationError):
        sm.complete_stage(3)


def test_artifact_existence_requirement(tmp_path):
    sm = QuantumStateMachine(root_dir=tmp_path)
    sm.init_project("test-project")

    # Mark stage 1 completed without creating the artifact
    # Note: complete_stage on 1 works, but stage 2 will fail if artifact is missing
    sm._data["stages"]["intent"]["status"] = "completed"
    sm.save()

    allowed, reason = sm.can_enter_stage(2)
    assert allowed is False
    assert "Artefato ausente" in reason

    # Now create the artifact
    art_path = tmp_path / INTENT_FILE
    art_path.parent.mkdir(parents=True, exist_ok=True)
    with open(art_path, "w", encoding="utf-8") as f:
        f.write("{}")

    allowed, reason = sm.can_enter_stage(2)
    assert allowed is True
    assert reason is None


def test_rollback(tmp_path):
    sm = QuantumStateMachine(root_dir=tmp_path)
    sm.init_project("test-project")

    # Mock stages 1, 2, 3 completed
    for s_name in ["intent", "env", "data-check"]:
        sm._data["stages"][s_name]["status"] = "completed"
    sm._data["current_stage"] = 4
    sm.save()

    assert sm.is_stage_completed("intent") is True
    assert sm.is_stage_completed("env") is True
    assert sm.is_stage_completed("data-check") is True

    # Rollback to stage 2
    sm.rollback_to(2)
    assert sm.current_stage_id == 2
    assert sm.is_stage_completed("intent") is True
    assert sm.is_stage_completed("env") is False
    assert sm.is_stage_completed("data-check") is False
