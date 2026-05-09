#!/usr/bin/env python3
"""
Sequential (referans), OpenMP ve CUDA image_processing ikililerini aynı girdi PGM ile çalıştırır,
stdout metriklerini kaydeder, çıktı PGMyleri Sequential ile karşılaştırır ve özet JSON + CSV üretir.

Linux CUDA laboratuvarı için tasarlanmıştır (RAPL için sequential/OMP sudo gerekebilir).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_tools import (
    append_metrics_csv,
    compare_pgm_files,
    parse_image_pipeline_stdout,
    write_json,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_cmd(
    argv: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        argv,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env={**dict(__import__("os").environ), **(env or {})},
    )


def stem_key(p: Path) -> str:
    return p.stem


def find_output_pair(cwd: Path) -> tuple[Path | None, Path | None]:
    blurs = sorted(cwd.glob("output_*_*_blurred.pgm"))
    edges = sorted(cwd.glob("output_*_*_edges.pgm"))
    return (blurs[0] if blurs else None, edges[0] if edges else None)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", errors="replace")


def main() -> int:
    rr = repo_root()

    ap = argparse.ArgumentParser(
        description="Image pipeline zamanlayıcısı + CPU/GPU çıktı karşılaştırması."
    )
    ap.add_argument(
        "--run-name",
        default="",
        help="Çalışma adı (boşsa UTC zaman damgası)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=rr / "scheduler" / "runs",
        help="Tüm çıktıların kökü (scheduler/runs)",
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Metrikleri eklemek için CSV (varsayılan: output-dir/current/metrics_append.csv yerine consolidated)",
    )
    ap.add_argument(
        "--image",
        type=Path,
        action="append",
        dest="images",
        help="Çalıştırılacak PGM (bir veya daha çok kez iletilebilir). Varsayılan: images/*.pgm",
    )
    ap.add_argument(
        "--seq-bin",
        type=Path,
        required=True,
        help="Sequential image_processing ikilisi",
    )
    ap.add_argument(
        "--omp-bin",
        type=Path,
        default=None,
        help="OpenMP image_processing_omp ikilisi (verilmezse atlanır)",
    )
    ap.add_argument(
        "--omp-threads",
        type=int,
        nargs="+",
        default=[8],
        help="OpenMP için iş parçacığı sayıları (birden fazla test için boşluktan ayrık liste)",
    )
    ap.add_argument(
        "--cuda-bin",
        type=Path,
        default=None,
        help="CUDA image_processing_cuda ikilisi (verilmezse atlanır)",
    )
    ap.add_argument(
        "--cuda-blocks",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="CUDA block boyutları (örn. 8 16 32)",
    )
    ap.add_argument(
        "--compare-threshold",
        type=int,
        default=3,
        help="compare_pgms için maks mutlak piksel farkı toleransı",
    )
    args = ap.parse_args()

    images = args.images
    if not images:
        imgs = sorted((rr / "images").glob("*.pgm"))
        images = imgs
        if not images:
            print("Hata: images/*.pgm bulunamadı; --image ile yol verin.", file=sys.stderr)
            return 2

    for im in images:
        if not im.exists():
            print(f"Hata: görüntü yok: {im}", file=sys.stderr)
            return 2

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root: Path = args.output_dir.resolve() / run_name
    consolidated_csv = (
        args.metrics_csv.resolve()
        if args.metrics_csv is not None
        else out_root / "metrics.csv"
    )

    seq_bin = args.seq_bin.resolve()
    omp_bin = args.omp_bin.resolve() if args.omp_bin else None
    cuda_bin = args.cuda_bin.resolve() if args.cuda_bin else None

    if not seq_bin.exists():
        print(f"Hata: sequential ikili yok: {seq_bin}", file=sys.stderr)
        return 2
    if omp_bin is not None and not omp_bin.exists():
        print(f"Hata: OpenMP ikili yok: {omp_bin}", file=sys.stderr)
        return 2
    if cuda_bin is not None and not cuda_bin.exists():
        print(f"Hata: CUDA ikili yok: {cuda_bin}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "run_name": run_name,
        "repo_root": str(rr),
        "sequence_binary": str(seq_bin),
        "openmp_binary": str(omp_bin) if omp_bin else "",
        "cuda_binary": str(cuda_bin) if cuda_bin else "",
        "compare_threshold_pixels": args.compare_threshold,
        "entries": [],
    }

    consolidated_csv.unlink(missing_ok=True)

    binary_fail_msgs: list[str] = []

    for img in images:
        img_abs = img.resolve()
        key = stem_key(img)

        entry_root: dict[str, Any] = {
            "image": str(img_abs),
            "image_stem": key,
            "variants": [],
        }

        def add_variant(kind: str, label: str, payload: dict[str, Any]) -> None:
            entry_root["variants"].append({"kind": kind, "label": label, **payload})

        ref_dir = out_root / "sequential" / key
        r = run_cmd([str(seq_bin), str(img_abs)], cwd=ref_dir)
        write_text(ref_dir / "stdout.txt", r.stdout or "")
        write_text(ref_dir / "stderr.txt", r.stderr or "")
        ref_rows = parse_image_pipeline_stdout(r.stdout or "")
        append_metrics_csv(consolidated_csv, run_name, key, ref_rows)
        b_ref, e_ref = find_output_pair(ref_dir)
        if b_ref is None or e_ref is None:
            print(
                f"Uyarı: sequential çıktıları bulunamadı ({key}), "
                "ikili doğru PGM üretmiş mi kontrol edin.",
                file=sys.stderr,
            )

        seq_payload = {
            "cwd": str(ref_dir),
            "returncode": r.returncode,
            "blur": str(b_ref) if b_ref else "",
            "edges": str(e_ref) if e_ref else "",
        }
        add_variant("sequential", "seq", seq_payload)

        if r.returncode != 0:
            binary_fail_msgs.append(
                f"{key} sequential returncode={r.returncode} (stderr bakın: {ref_dir / 'stderr.txt'})"
            )

        if omp_bin and r.returncode == 0 and b_ref and e_ref:
            for nt in args.omp_threads:
                cwd = out_root / "openmp" / f"t{int(nt)}" / key
                rr_omp = run_cmd([str(omp_bin), str(img_abs), str(int(nt))], cwd=cwd)
                write_text(cwd / "stdout.txt", rr_omp.stdout or "")
                write_text(cwd / "stderr.txt", rr_omp.stderr or "")
                rows = parse_image_pipeline_stdout(rr_omp.stdout or "")
                append_metrics_csv(consolidated_csv, run_name, key, rows)
                b_cmp, e_cmp = find_output_pair(cwd)

                compares: dict[str, Any] = {}
                if b_cmp and b_ref:
                    c0 = compare_pgm_files(b_ref.resolve(), b_cmp.resolve(), threshold=args.compare_threshold)
                    compares["blurred_vs_sequential"] = asdict(c0)
                else:
                    compares["blurred_vs_sequential"] = {"error": "missing_output"}

                if e_cmp and e_ref:
                    c1 = compare_pgm_files(e_ref.resolve(), e_cmp.resolve(), threshold=args.compare_threshold)
                    compares["edges_vs_sequential"] = asdict(c1)
                else:
                    compares["edges_vs_sequential"] = {"error": "missing_output"}

                if rr_omp.returncode != 0:
                    binary_fail_msgs.append(
                        f"{key} openmp t={int(nt)} returncode={rr_omp.returncode}"
                    )

                payload = {
                    "cwd": str(cwd),
                    "threads": int(nt),
                    "returncode": rr_omp.returncode,
                    "blur": str(b_cmp) if b_cmp else "",
                    "edges": str(e_cmp) if e_cmp else "",
                    "comparison": compares,
                }
                add_variant("openmp", f"t{int(nt)}", payload)

        if cuda_bin and r.returncode == 0 and b_ref and e_ref:
            for blk in args.cuda_blocks:
                if int(blk) not in (8, 16, 32):
                    print(
                        f"Uyarı: blok {blk} beklenen kümede değil (8,16,32); CUDA ikili reddedebilir.",
                        file=sys.stderr,
                    )
                cwd = out_root / "cuda" / f"b{int(blk)}" / key
                rr_cu = run_cmd([str(cuda_bin), str(img_abs), str(int(blk))], cwd=cwd)
                write_text(cwd / "stdout.txt", rr_cu.stdout or "")
                write_text(cwd / "stderr.txt", rr_cu.stderr or "")
                rows = parse_image_pipeline_stdout(rr_cu.stdout or "")
                append_metrics_csv(consolidated_csv, run_name, key, rows)
                b_cmp, e_cmp = find_output_pair(cwd)

                compares: dict[str, Any] = {}
                if b_cmp and b_ref:
                    c0 = compare_pgm_files(b_ref.resolve(), b_cmp.resolve(), threshold=args.compare_threshold)
                    compares["blurred_vs_sequential"] = asdict(c0)
                else:
                    compares["blurred_vs_sequential"] = {"error": "missing_output"}

                if e_cmp and e_ref:
                    c1 = compare_pgm_files(e_ref.resolve(), e_cmp.resolve(), threshold=args.compare_threshold)
                    compares["edges_vs_sequential"] = asdict(c1)
                else:
                    compares["edges_vs_sequential"] = {"error": "missing_output"}

                if rr_cu.returncode != 0:
                    binary_fail_msgs.append(
                        f"{key} cuda block={int(blk)} returncode={rr_cu.returncode}"
                    )

                payload = {
                    "cwd": str(cwd),
                    "cuda_block": int(blk),
                    "returncode": rr_cu.returncode,
                    "blur": str(b_cmp) if b_cmp else "",
                    "edges": str(e_cmp) if e_cmp else "",
                    "comparison": compares,
                }
                add_variant("cuda", f"b{int(blk)}", payload)

        summary["entries"].append(entry_root)

    write_json(out_root / "summary.json", summary)

    failures: list[str] = []
    ok: list[str] = []
    for ent in summary["entries"]:
        for v in ent.get("variants", []):
            if v.get("kind") in ("openmp", "cuda"):
                comp = v.get("comparison") or {}
                for name, payload in comp.items():
                    if isinstance(payload, dict) and payload.get("error"):
                        failures.append(
                            f"{ent.get('image_stem')} {v.get('kind')}:{v.get('label')} {name} [{payload['error']}]"
                        )
                    elif isinstance(payload, dict) and payload.get("ok") is False:
                        failures.append(
                            f"{ent.get('image_stem')} {v.get('kind')}:{v.get('label')} {name}"
                        )
                    elif isinstance(payload, dict) and payload.get("ok") is True:
                        ok.append(
                            f"{ent.get('image_stem')} {v.get('kind')}:{v.get('label')} {name}"
                        )

    print("")
    print(f"Çıktılar:       {out_root}")
    print(f"Metrik CSV:      {consolidated_csv}")
    print(f"Özet JSON:       {out_root / 'summary.json'}")
    print(f"Tolerans (px):   {args.compare_threshold}")
    print("")
    print(f"Uyumlu kıyas: {len(ok)}")
    print(f"Hatalı kıyas / eksik çıktı: {len(failures)}")
    if failures:
        for fline in failures:
            print("  FAIL →", fline)
        print("", file=sys.stderr)
        print("Tolerans veya eksik PGM nedeniyle hata var.", file=sys.stderr)

    if binary_fail_msgs:
        print("", file=sys.stderr)
        print("İkili hataları:", file=sys.stderr)
        for m in binary_fail_msgs:
            print("  BINARY →", m, file=sys.stderr)

    exit_code = 0
    if failures:
        exit_code = max(exit_code, 1)
    if binary_fail_msgs:
        exit_code = max(exit_code, 2)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
