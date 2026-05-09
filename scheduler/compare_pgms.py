#!/usr/bin/env python3
"""İki PGM (P5) dosyasını piksel piksel karşılaştırır (CPU/GPU doğruluğu için)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_tools import compare_pgm_files


def main() -> int:
    p = argparse.ArgumentParser(description="İki PGMyi karşılaştırır (P5 / maxval 255).")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Maks mutlak fark bu değeri aşmayanlar 'uyumlu' kabul edilir (varsayılan: 3).",
    )
    args = p.parse_args()

    if not args.a.exists() or not args.b.exists():
        print("Hata: dosyalardan biri bulunamadı.", file=sys.stderr)
        return 2

    res = compare_pgm_files(args.a, args.b, threshold=args.threshold)
    print(f"A:           {res.path_a}")
    print(f"B:           {res.path_b}")
    print(f"Boyut:       {res.width}x{res.height}")
    print(f"Max |fark|:  {res.max_abs_diff}")
    print(f"Ort |fark|:  {res.mean_abs_diff:.6f}")
    print(f"RMSE:        {res.rmse:.6f}")
    print(f"> eşik piksel: {res.differing_pixels} (eşik={res.threshold})")
    print(f"Sonuç:       {'OK' if res.ok else 'FAIL'}")

    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
