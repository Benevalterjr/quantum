"""
Tests for Quantum Hardware Inspector & Cloud Profiler.
"""
from pathlib import Path
from quantum.hardware import HardwareInspector


def test_hardware_inspector_local(tmp_path):
    inspector = HardwareInspector(root_dir=tmp_path)
    spec = inspector.profile_environment(target="local")
    assert "os" in spec
    assert "cpu_cores" in spec
    assert "system_ram_gb" in spec
    assert spec["system_ram_gb"] > 0
    assert "recommended_quantization" in spec


def test_hardware_inspector_colab(tmp_path):
    inspector = HardwareInspector(root_dir=tmp_path)
    spec = inspector.profile_environment(target="colab", cloud_profile="colab-free")
    assert spec["target"] == "colab"
    assert spec["vram_gb"] == 15.0
    assert spec["recommended_quantization"] == "4bit"
    assert spec["recommended_batch_size"] == 2


def test_hardware_inspector_runpod(tmp_path):
    inspector = HardwareInspector(root_dir=tmp_path)
    spec = inspector.profile_environment(target="runpod", cloud_profile="runpod-4090")
    assert spec["vram_gb"] == 24.0
    assert spec["bf16_supported"] is True


def test_save_environment_file(tmp_path):
    inspector = HardwareInspector(root_dir=tmp_path)
    spec = inspector.profile_environment(target="colab")
    p = inspector.save_environment(spec)
    assert p.exists()
    assert (tmp_path / ".quantum" / "environment.json").exists()
