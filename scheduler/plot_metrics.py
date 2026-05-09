#!/usr/bin/env python3
"""
metrics.csv (run_benchmark çıktısı) üzerinden GFLOPS ve GFLOPS/W çubuk grafikleri üretir.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _float_or_none(raw: str) -> float | None:
    raw = (raw or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _display_name(row: dict[str, str]) -> str:
    b = (row.get("backend") or "").strip().lower()
    if b == "openmp":
        return f"OMP-{row.get('threads', '').strip() or '?'}"
    if b == "cuda":
        return f"CUDA-{row.get('block', '').strip() or '?'}"
    if b == "sequential":
        return "SEQ"
    lbl = row.get("label") or ""
    return lbl.strip() if lbl.strip() else b or "?"


def _load_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fp:
        rdr = csv.DictReader(fp)
        for row in rdr:
            rows.append(dict(row))
    return rows


def _plot_metric(
    rows: list[dict[str, str]],
    *,
    image_stem: str,
    phase: str,
    metric_key: str,
    metric_title: str,
    out_file: Path,
) -> bool:
    subset = [
        r
        for r in rows
        if r.get("image_stem") == image_stem and r.get("phase") == phase
    ]
    if not subset:
        return False

    order: dict[str, float] = {}
    ordering_rank = defaultdict(lambda: 1_000_000.0)

    rank_map = {"sequential": 0, "openmp": 1, "cuda": 2, "unknown": 9}

    for r in subset:
        lbl = _display_name(r)
        val = _float_or_none(r.get(metric_key, ""))
        if val is None:
            continue
        bk = (r.get("backend") or "unknown").lower()
        rk = rank_map.get(bk, 5)

        omp_t = float(r.get("threads") or 0)
        cuda_b = float(r.get("block") or 0)

        extra = omp_t / 1000.0 + cuda_b / 1_000_000.0
        ordering_rank[lbl] = rk + extra
        order[lbl] = val

    if not order:
        return False

    labels_sorted = sorted(order.keys(), key=lambda lbl: ordering_rank[lbl])
    vals = [order[l] for l in labels_sorted]

    plt.figure(figsize=(max(10, 1.6 * len(labels_sorted)), 5.5))
    plt.bar(labels_sorted, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric_title)
    plt.title(f"{metric_title}\n({image_stem} · {phase})")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=160)
    plt.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="metrics.csv için GFLOPS / GFLOPS/W grafikleri.")
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="run_benchmark çıktı metrics.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "graphs",
        help="PNGlerin yazılacağı klasör",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"Hata: CSV yok: {args.csv}")
        return 2

    rows = _load_csv(args.csv)
    images = sorted({r.get("image_stem") for r in rows if r.get("image_stem")})
    phases = ["gaussian", "sobel"]

    out_dir = args.output_dir.resolve()
    count = 0
    for im in images:
        for phase in phases:
            for stem, metric, title in (
                ("gflops", "gflops", "GFLOPS"),
                ("gflops_w", "gflops_w", "GFLOPS/W"),
            ):
                out = out_dir / f"{stem}_{phase}_{im}.png"
                if _plot_metric(
                    rows,
                    image_stem=im,
                    phase=phase,
                    metric_key=metric,
                    metric_title=title,
                    out_file=out,
                ):
                    count += 1

    print(f"{count} grafik olusturuldu: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
