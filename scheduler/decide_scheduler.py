#!/usr/bin/env python3
"""
Checkpoint §4 güç bütçesine göre CPU vs GPU seçimini yazdırır veya çıkış kodu olarak döner (--exit-code).

Örnek:
  python decide_scheduler.py --budget 150 --cpu-energy 12 --gpu-energy 9 --gpu-power 120

Canlı tek örnek GPU gücü (Linux + nvidia-smi):
  python decide_scheduler.py --budget 100 --cpu-energy 8 --gpu-energy 5 --poll-gpu
"""

from __future__ import annotations

import argparse
import sys

from energy_policy import choose_device, sample_gpu_power_draw_watts


def main() -> int:
    ap = argparse.ArgumentParser(description="Checkpoint §4 enerji/güç bütçesi cihaz seçimi.")
    ap.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Güç üst sinir (W), ör. checkpoint: 75, 100 veya 150",
    )
    ap.add_argument(
        "--cpu-energy",
        type=float,
        default=None,
        help="Ölçülen CPU iş toplami enerji (Joule)",
    )
    ap.add_argument(
        "--gpu-energy",
        type=float,
        default=None,
        help="Ölçülen GPU iş toplami enerji (Joule)",
    )
    ap.add_argument(
        "--gpu-power",
        type=float,
        default=None,
        help="GPU power.draw için temsil gücü (W); --poll-gpu verilirse görmezden gelinabilir",
    )
    ap.add_argument(
        "--poll-gpu",
        action="store_true",
        help="gpu-power yerine tek nvidia-smi örneği kullanır",
    )
    ap.add_argument(
        "--exit-code",
        action="store_true",
        help="Çıkış: GPU seçildiyse 0, CPU seçildiyse 1",
    )
    args = ap.parse_args()

    gpu_w = args.gpu_power
    if args.poll_gpu:
        gpu_w = sample_gpu_power_draw_watts()
        if gpu_w is None:
            print(
                "Uyar.: nvidia-smi ile GPU güc okunamadı (--gpu-power manuel verin).",
                file=sys.stderr,
            )

    result = choose_device(
        cpu_energy_j=args.cpu_energy,
        gpu_energy_j=args.gpu_energy,
        gpu_power_w=gpu_w,
        budget_w=args.budget,
    )

    print(f"Cihaz: {result.device.upper()}")
    print(f"Gerekçe: {result.reason}")

    if args.exit_code:
        return 0 if result.device == "gpu" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
