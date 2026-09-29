"""
Hardware & Environment Profiler for the Quantum Framework.
Detects GPU, VRAM, CPU, RAM, and recommends execution parameters.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from quantum.constants import ENV_FILE, QUANTUM_DIR


CLOUD_PROFILES = {
    "colab-free": {
        "name": "Google Colab (Free T4)",
        "gpu_name": "NVIDIA Tesla T4",
        "vram_gb": 15.0,
        "compute_capability": "7.5",
        "bf16_supported": False,
        "recommended_quantization": "4bit",
        "recommended_batch_size": 2,
        "recommended_grad_accum": 4,
    },
    "colab-pro": {
        "name": "Google Colab Pro (A100)",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "vram_gb": 40.0,
        "compute_capability": "8.0",
        "bf16_supported": True,
        "recommended_quantization": "4bit",
        "recommended_batch_size": 4,
        "recommended_grad_accum": 2,
    },
    "runpod-4090": {
        "name": "RunPod (RTX 4090)",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "vram_gb": 24.0,
        "compute_capability": "8.9",
        "bf16_supported": True,
        "recommended_quantization": "4bit",
        "recommended_batch_size": 4,
        "recommended_grad_accum": 2,
    },
    "runpod-a6000": {
        "name": "RunPod (RTX A6000)",
        "gpu_name": "NVIDIA RTX A6000",
        "vram_gb": 48.0,
        "compute_capability": "8.6",
        "bf16_supported": True,
        "recommended_quantization": "16bit",
        "recommended_batch_size": 8,
        "recommended_grad_accum": 2,
    },
}


class HardwareInspector:
    def __init__(self, root_dir: Path | str = "."):
        self.root_dir = Path(root_dir)

    def detect_system_ram_gb(self) -> float:
        """Detect system physical RAM in GB."""
        try:
            if platform.system() == "Windows":
                # PowerShell query
                cmd = "Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object -ExpandProperty Sum"
                output = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, stderr=subprocess.DEVNULL)
                val = int(output.strip())
                return round(val / (1024 ** 3), 2)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            kb = int(parts[1])
                            return round(kb / (1024 ** 2), 2)
            elif platform.system() == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                return round(int(out.strip()) / (1024 ** 3), 2)
        except Exception:
            pass
        return 16.0  # safe default fallback

    def detect_nvidia_gpu(self) -> Optional[Dict[str, Any]]:
        """Inspect NVIDIA GPU using PyTorch if installed or nvidia-smi."""
        # 1. Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = 0
                props = torch.cuda.get_device_properties(device_idx)
                total_vram_gb = round(props.total_memory / (1024 ** 3), 2)
                cap = f"{props.major}.{props.minor}"
                bf16 = props.major >= 8
                return {
                    "source": "torch.cuda",
                    "gpu_name": props.name,
                    "gpu_count": torch.cuda.device_count(),
                    "vram_gb": total_vram_gb,
                    "compute_capability": cap,
                    "bf16_supported": bf16,
                }
        except Exception:
            pass

        # 2. Try nvidia-smi
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            lines = out.strip().splitlines()
            if lines:
                parts = lines[0].split(",")
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return {
                    "source": "nvidia-smi",
                    "gpu_name": name,
                    "gpu_count": len(lines),
                    "vram_gb": round(vram_mb / 1024, 2),
                    "compute_capability": "unknown",
                    "bf16_supported": False,
                }
        except Exception:
            pass

        return None

    def profile_environment(
        self,
        target: str = "local",
        cloud_profile: Optional[str] = None,
        custom_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Profiles the training environment and produces environment specs.
        """
        sys_ram = self.detect_system_ram_gb()
        cpu_cores = os.cpu_count() or 4
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        gpu_info = self.detect_nvidia_gpu()

        if target == "local":
            if gpu_info is not None:
                vram = custom_vram_gb or gpu_info["vram_gb"]
                name = gpu_info["gpu_name"]
                bf16 = gpu_info["bf16_supported"]
                cap = gpu_info["compute_capability"]
                mode = "local_gpu"
            else:
                # Local CPU fallback or custom VRAM
                vram = custom_vram_gb or 0.0
                name = "Local CPU (No NVIDIA GPU detected)"
                bf16 = False
                cap = "N/A"
                mode = "local_cpu"
        elif target in ("colab", "runpod", "cloud"):
            profile_key = cloud_profile or ("colab-free" if target == "colab" else "runpod-4090")
            profile = CLOUD_PROFILES.get(profile_key, CLOUD_PROFILES["colab-free"])
            vram = custom_vram_gb or profile["vram_gb"]
            name = profile["gpu_name"]
            bf16 = profile["bf16_supported"]
            cap = profile["compute_capability"]
            mode = f"cloud_{target}"
        else:
            vram = custom_vram_gb or 16.0
            name = f"Custom ({target})"
            bf16 = True
            cap = "8.0"
            mode = "custom"

        # Calculate recommendations
        if vram >= 40.0:
            recommended_quant = "16bit"
            recommended_batch_size = 4
            grad_accum = 2
            max_seq = 4096
        elif vram >= 15.0:
            recommended_quant = "4bit"
            recommended_batch_size = 2
            grad_accum = 4
            max_seq = 2048
        elif vram >= 8.0:
            recommended_quant = "4bit"
            recommended_batch_size = 1
            grad_accum = 8
            max_seq = 1024
        else:
            recommended_quant = "4bit"
            recommended_batch_size = 1
            grad_accum = 16
            max_seq = 512

        env_spec = {
            "target": target,
            "mode": mode,
            "os": os_info,
            "python_version": platform.python_version(),
            "cpu_cores": cpu_cores,
            "system_ram_gb": sys_ram,
            "gpu_name": name,
            "vram_gb": vram,
            "compute_capability": cap,
            "bf16_supported": bf16,
            "recommended_quantization": recommended_quant,
            "recommended_batch_size": recommended_batch_size,
            "recommended_gradient_accumulation_steps": grad_accum,
            "recommended_max_seq_length": max_seq,
            "notes": (
                "Para GPUs com menos de 16GB VRAM, recomenda-se fortemente o uso de QLoRA 4-bit (NF4) "
                "com Unsloth ou bitsandbytes e gradient checkpointing."
            ),
        }
        return env_spec

    def save_environment(self, env_spec: Dict[str, Any]) -> Path:
        target_path = self.root_dir / ENV_FILE
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(env_spec, f, indent=2, ensure_ascii=False)
        return target_path
