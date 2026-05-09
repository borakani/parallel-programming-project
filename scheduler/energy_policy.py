"""
Checkpoint enerji-duyarlı zamanlayici (§4): olcu edilen enerji ve güc budget'ina göre cihaz secimi.

Kural:
  GPU secilir eger  (gpu_energy_j < cpu_energy_j) VE (gpu_power_w < budget_w)
  degilse CPU.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Literal

Device = Literal["gpu", "cpu"]


@dataclass(frozen=True)
class PolicyResult:
    device: Device
    reason: str


def choose_device(
    cpu_energy_j: float | None,
    gpu_energy_j: float | None,
    gpu_power_w: float | None,
    budget_w: float,
) -> PolicyResult:
    """
    Enerji degerleri ayni is yuku (or. gaussian) icin toplam Joule olmalı.
    gpu_power_w: nvidia-smi'den orneklenmis ortalama cekiş gücü (W).
    """
    if cpu_energy_j is None or gpu_energy_j is None:
        return PolicyResult(
            "cpu",
            "CPU veya GPU enerjisi olculmemis (NA); güvenli varsayım: CPU.",
        )

    if budget_w <= 0:
        return PolicyResult("cpu", "Güç bütçesi > 0 degil.")

    if gpu_power_w is None or not math.isfinite(gpu_power_w) or gpu_power_w <= 0:
        return PolicyResult(
            "cpu",
            "GPU güc ölçümü güvenilir degil; budget karsi lastirmasi yapilmadi.",
        )

    gpu_better_energy = gpu_energy_j < cpu_energy_j
    within_budget = gpu_power_w < budget_w

    if gpu_better_energy and within_budget:
        return PolicyResult(
            "gpu",
            f"gpu_energy ({gpu_energy_j:.4g} J) < cpu_energy ({cpu_energy_j:.4g} J) ve "
            f"gpu_power ({gpu_power_w:.4g} W) < budget ({budget_w:.4g} W).",
        )

    parts = []
    if not gpu_better_energy:
        parts.append(
            f"GPU enerjisi daha yüksek veya esit ({gpu_energy_j:.4g} J >= {cpu_energy_j:.4g} J)"
        )
    if not within_budget:
        parts.append(
            f"GPU gücü budget üstünde veya esit ({gpu_power_w:.4g} W >= {budget_w:.4g} W)"
        )
    return PolicyResult("cpu", " ; ".join(parts) + ".")


def sample_gpu_power_draw_watts() -> float | None:
    """nvidia-smi ile tek örnek (W). Yoksa None."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first = proc.stdout.strip().splitlines()[0].strip().split(",")[0].strip()
        val = float(first)
        if val <= 0:
            return None
        return val
    except (OSError, ValueError):
        return None
