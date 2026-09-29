"""
State Machine for the Quantum 11-Stage LLM Harness.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantum.constants import (
    QUANTUM_DIR,
    STATE_FILE,
    STAGES,
    STAGE_NAME_TO_ID,
    STAGE_ID_TO_NAME,
)


class StageValidationError(Exception):
    """Raised when a stage transition or validation fails."""
    pass


class QuantumStateMachine:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)
        self.quantum_dir = self.root_dir / QUANTUM_DIR
        self.state_file = self.root_dir / STATE_FILE
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load state from state.json or initialize if missing."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                    return
            except Exception:
                pass
        self._data = self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "project_name": "quantum-project",
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "current_stage": 1,
            "stages": {
                stage["name"]: {
                    "id": stage["id"],
                    "title": stage["title"],
                    "status": "pending",  # pending, in_progress, completed, failed
                    "completed_at": None,
                    "artifact": stage["artifact"],
                    "metadata": {},
                }
                for stage in STAGES
            },
        }

    def save(self) -> None:
        """Persist state to .quantum/state.json."""
        self.quantum_dir.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def init_project(self, project_name: str = "quantum-project", force: bool = False) -> None:
        """Initialize or reset a Quantum project."""
        self.quantum_dir.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists() and not force:
            return
        self._data = self._default_state()
        self._data["project_name"] = project_name
        self.save()

    @property
    def current_stage_id(self) -> int:
        return self._data.get("current_stage", 1)

    @property
    def project_name(self) -> str:
        return self._data.get("project_name", "quantum-project")

    def get_stage_info(self, stage_ref: int | str) -> Dict[str, Any]:
        """Get stage info by id or name."""
        stage_name = STAGE_ID_TO_NAME[stage_ref] if isinstance(stage_ref, int) else stage_ref
        if stage_name not in self._data["stages"]:
            raise StageValidationError(f"Estágio desconhecido: {stage_ref}")
        return self._data["stages"][stage_name]

    def is_stage_completed(self, stage_ref: int | str) -> bool:
        info = self.get_stage_info(stage_ref)
        return info["status"] == "completed"

    def can_enter_stage(self, target_stage_ref: int | str) -> tuple[bool, Optional[str]]:
        """
        Check if prerequisite stages are completed.
        Strict sequential harness rule: Cannot run Stage N without Stage 1..N-1 completed.
        """
        target_name = STAGE_ID_TO_NAME[target_stage_ref] if isinstance(target_stage_ref, int) else target_stage_ref
        target_id = STAGE_NAME_TO_ID[target_name]

        if target_id == 1:
            return True, None

        # Check all previous stages
        for prev_id in range(1, target_id):
            prev_name = STAGE_ID_TO_NAME[prev_id]
            prev_info = self._data["stages"][prev_name]
            if prev_info["status"] != "completed":
                return False, (
                    f"Bloqueio de Harness: A etapa {target_id} ({target_name}) requer "
                    f"que a etapa {prev_id} ({prev_name}) seja concluída com sucesso antes!"
                )
            # Verify artifact exists on disk
            artifact_rel = prev_info["artifact"]
            artifact_path = self.root_dir / artifact_rel
            if not artifact_path.exists():
                return False, (
                    f"Artefato ausente: O arquivo esperado para a etapa {prev_id} "
                    f"({artifact_rel}) não foi encontrado no disco!"
                )

        return True, None

    def complete_stage(self, stage_ref: int | str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark stage as completed, update metadata and advance current_stage pointer."""
        stage_name = STAGE_ID_TO_NAME[stage_ref] if isinstance(stage_ref, int) else stage_ref
        stage_id = STAGE_NAME_TO_ID[stage_name]

        allowed, reason = self.can_enter_stage(stage_id)
        if not allowed:
            raise StageValidationError(reason)

        stage_data = self._data["stages"][stage_name]
        stage_data["status"] = "completed"
        stage_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        if metadata:
            stage_data["metadata"].update(metadata)

        # Advance current stage to next if not at end
        if stage_id < 11:
            self._data["current_stage"] = max(self._data["current_stage"], stage_id + 1)
        self.save()

    def rollback_to(self, target_stage_ref: int | str) -> None:
        """
        Rollback to a specific stage, resetting this and all subsequent stages to pending.
        """
        target_name = STAGE_ID_TO_NAME[target_stage_ref] if isinstance(target_stage_ref, int) else target_stage_ref
        target_id = STAGE_NAME_TO_ID[target_name]

        for s_id in range(target_id, 12):
            s_name = STAGE_ID_TO_NAME[s_id]
            self._data["stages"][s_name]["status"] = "pending"
            self._data["stages"][s_name]["completed_at"] = None

        self._data["current_stage"] = target_id
        self.save()

    def get_summary_table(self) -> List[Dict[str, Any]]:
        """Generate structured data for status reporting."""
        rows = []
        for s in STAGES:
            s_data = self._data["stages"][s["name"]]
            artifact_path = self.root_dir / s["artifact"]
            rows.append({
                "id": s["id"],
                "name": s["name"],
                "title": s["title"],
                "status": s_data["status"],
                "artifact": s["artifact"],
                "artifact_exists": artifact_path.exists(),
                "completed_at": s_data.get("completed_at"),
                "metadata": s_data.get("metadata", {}),
            })
        return rows
