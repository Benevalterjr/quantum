"""
Tests for Quantum Delivery & Packaging (Modelfile, vLLM, HANDOVER).
"""
import json
from pathlib import Path
from quantum.delivery import Deliverer
from quantum.constants import MODELFILE_PATH, HANDOVER_FILE, DELIVERY_SPEC_FILE


def test_delivery_packaging(tmp_path):
    deliverer = Deliverer(root_dir=tmp_path)
    res = deliverer.prepare_delivery(service_target="ollama", model_tag="quantum-juridico:v1")

    assert res["status"] == "READY_FOR_DEPLOYMENT"
    assert res["model_tag"] == "quantum-juridico:v1"

    modelfile = tmp_path / MODELFILE_PATH
    vllm_sh = tmp_path / "serve_vllm.sh"
    serve_api = tmp_path / "serve_api.py"
    handover = tmp_path / HANDOVER_FILE
    spec_json = tmp_path / DELIVERY_SPEC_FILE

    assert modelfile.exists()
    assert vllm_sh.exists()
    assert serve_api.exists()
    assert handover.exists()
    assert spec_json.exists()

    # Verify Modelfile contents
    m_content = modelfile.read_text(encoding="utf-8")
    assert "FROM ./export/" in m_content
    assert "PARAMETER temperature" in m_content
    assert "TEMPLATE" in m_content

    # Verify Handover contents
    h_content = handover.read_text(encoding="utf-8")
    assert "ollama create" in h_content
    assert "vLLM" in h_content
