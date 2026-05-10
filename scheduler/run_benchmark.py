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
    average_parsed_metric_rows,
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


def exec_pipeline_repeated(
    argv: list[str],
    cwd: Path,
    *,
    repeats: int,
    stdout_stem: str = "stdout",
) -> tuple[
    subprocess.CompletedProcess[str] | None,
    list[dict[str, Any]],
    list[str],
    int,
]:
    """
    Ikiliyi `repeats` kez aynı cwd icinde calistirir.
    CSV: tum basarili ayristirmalar faz bazinda ortalanir (`samples`).
    Son donus: CompletedProcess?, metrik satirlari, uyari metinleri,
    basarili tekrar sayisi (tam basari icin repeats ile esit olmali).
    """
    notes: list[str] = []
    good_parses: list[list[dict[str, Any]]] = []
    last_cp: subprocess.CompletedProcess[str] | None = None

    if repeats < 1:
        return None, [], ["repeat < 1"], 0

    for idx in range(repeats):
        last_cp = run_cmd(argv, cwd=cwd)
        write_text(cwd / f"{stdout_stem}_{idx + 1:02d}.txt", last_cp.stdout or "")
        write_text(cwd / f"{stdout_stem}_{idx + 1:02d}.stderr.txt", last_cp.stderr or "")
        write_text(cwd / "stderr.txt", last_cp.stderr or "")
        if last_cp.returncode != 0:
            notes.append(f"Tekrar {idx + 1}/{repeats} rc={last_cp.returncode}")
            continue
        parsed = parse_image_pipeline_stdout(last_cp.stdout or "")
        if not parsed:
            notes.append(f"Tekrar {idx + 1}/{repeats} stdout ayristiramadi")
            continue
        good_parses.append(parsed)

    if last_cp is not None:
        write_text(cwd / "stdout.txt", last_cp.stdout or "")

    n_ok = len(good_parses)
    if not good_parses:
        return last_cp, [], notes, 0

    if n_ok < repeats:
        notes.append(
            f"Basarili tekrar: {n_ok}/{repeats} "
            "(ortalama yalnizca basarili tekrarlara gore)."
        )

    aggregated = average_parsed_metric_rows(good_parses)
    if not aggregated:
        notes.append("Faz uyumsuzu; ortalama yok, son basarili parse CSVye yazilir.")
        fb = [{**r} for r in good_parses[-1]]
        for row in fb:
            row["samples"] = str(n_ok)
        return last_cp, fb, notes, n_ok

    return last_cp, aggregated, notes, n_ok


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
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Her ikili konfig için tekrarlama (Checkpoint:≥5 için N=5 kullanın); metrik CSV ortalamasi yazilir.",
    )
    args = ap.parse_args()

    if args.repeat < 1:
        print("Hata: --repeat en az 1 olmalidir.", file=sys.stderr)
        return 2

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
        r, ref_rows, seq_rep_notes, n_ok_seq = exec_pipeline_repeated(
            [str(seq_bin), str(img_abs)],
            ref_dir,
            repeats=args.repeat,
            stdout_stem="stdout",
        )
        for n in seq_rep_notes:
            print(f"Uyarı (sequential {key}): {n}", file=sys.stderr)
        append_metrics_csv(consolidated_csv, run_name, key, ref_rows)
        b_ref, e_ref = find_output_pair(ref_dir)
        if b_ref is None or e_ref is None:
            print(
                f"Uyarı: sequential çıktıları bulunamadı ({key}), "
                "ikili doğru PGM üretmiş mi kontrol edin.",
                file=sys.stderr,
            )

        seq_rc = r.returncode if r is not None else -1
        seq_payload = {
            "cwd": str(ref_dir),
            "returncode": seq_rc,
            "repeat": args.repeat,
            "successful_repeats": n_ok_seq,
            "blur": str(b_ref) if b_ref else "",
            "edges": str(e_ref) if e_ref else "",
        }
        add_variant("sequential", "seq", seq_payload)

        if (
            r is None
            or r.returncode != 0
            or n_ok_seq != args.repeat
            or not ref_rows
        ):
            binary_fail_msgs.append(
                f"{key} sequential rc={seq_rc} tekrarlar={n_ok_seq}/{args.repeat} "
                f"(stderr: {ref_dir / 'stderr.txt'})"
            )

        seq_ok = (
            r is not None
            and r.returncode == 0
            and n_ok_seq == args.repeat
            and len(ref_rows) > 0
            and b_ref is not None
            and e_ref is not None
        )

        if omp_bin and seq_ok and b_ref and e_ref:
            for nt in args.omp_threads:
                cwd = out_root / "openmp" / f"t{int(nt)}" / key
                rr_omp, rows, omp_notes, n_ok_omp = exec_pipeline_repeated(
                    [str(omp_bin), str(img_abs), str(int(nt))],
                    cwd,
                    repeats=args.repeat,
                    stdout_stem="stdout",
                )
                for n in omp_notes:
                    print(f"Uyarı (openmp t={nt} {key}): {n}", file=sys.stderr)
                append_metrics_csv(consolidated_csv, run_name, key, rows)
                b_cmp, e_cmp = find_output_pair(cwd)

                compares: dict[str, Any] = {}
                rep_ok = n_ok_omp == args.repeat and bool(rows)
                if rep_ok and b_cmp and b_ref:
                    c0 = compare_pgm_files(b_ref.resolve(), b_cmp.resolve(), threshold=args.compare_threshold)
                    compares["blurred_vs_sequential"] = asdict(c0)
                else:
                    err = "missing_output"
                    if args.repeat > 1 and not rep_ok:
                        err = "incomplete_repeats_or_parse"
                    compares["blurred_vs_sequential"] = {"error": err}

                if rep_ok and e_cmp and e_ref:
                    c1 = compare_pgm_files(e_ref.resolve(), e_cmp.resolve(), threshold=args.compare_threshold)
                    compares["edges_vs_sequential"] = asdict(c1)
                else:
                    err = "missing_output"
                    if args.repeat > 1 and not rep_ok:
                        err = "incomplete_repeats_or_parse"
                    compares["edges_vs_sequential"] = {"error": err}

                omp_rc = rr_omp.returncode if rr_omp is not None else -1
                if (
                    rr_omp is None
                    or omp_rc != 0
                    or n_ok_omp != args.repeat
                    or not rows
                ):
                    binary_fail_msgs.append(
                        f"{key} openmp t={int(nt)} rc={omp_rc} tekrarlar={n_ok_omp}/{args.repeat}"
                    )

                payload = {
                    "cwd": str(cwd),
                    "threads": int(nt),
                    "returncode": omp_rc,
                    "repeat": args.repeat,
                    "successful_repeats": n_ok_omp,
                    "blur": str(b_cmp) if b_cmp else "",
                    "edges": str(e_cmp) if e_cmp else "",
                    "comparison": compares,
                }
                add_variant("openmp", f"t{int(nt)}", payload)

        if cuda_bin and seq_ok and b_ref and e_ref:
            for blk in args.cuda_blocks:
                if int(blk) not in (8, 16, 32):
                    print(
                        f"Uyarı: blok {blk} beklenen kümede değil (8,16,32); CUDA ikili reddedebilir.",
                        file=sys.stderr,
                    )
                cwd = out_root / "cuda" / f"b{int(blk)}" / key
                rr_cu, rows, cuda_notes, n_ok_cu = exec_pipeline_repeated(
                    [str(cuda_bin), str(img_abs), str(int(blk))],
                    cwd,
                    repeats=args.repeat,
                    stdout_stem="stdout",
                )
                for n in cuda_notes:
                    print(f"Uyarı (cuda b={blk} {key}): {n}", file=sys.stderr)
                append_metrics_csv(consolidated_csv, run_name, key, rows)
                b_cmp, e_cmp = find_output_pair(cwd)

                compares: dict[str, Any] = {}
                rep_ok = n_ok_cu == args.repeat and bool(rows)
                if rep_ok and b_cmp and b_ref:
                    c0 = compare_pgm_files(b_ref.resolve(), b_cmp.resolve(), threshold=args.compare_threshold)
                    compares["blurred_vs_sequential"] = asdict(c0)
                else:
                    err = "missing_output"
                    if args.repeat > 1 and not rep_ok:
                        err = "incomplete_repeats_or_parse"
                    compares["blurred_vs_sequential"] = {"error": err}

                if rep_ok and e_cmp and e_ref:
                    c1 = compare_pgm_files(e_ref.resolve(), e_cmp.resolve(), threshold=args.compare_threshold)
                    compares["edges_vs_sequential"] = asdict(c1)
                else:
                    err = "missing_output"
                    if args.repeat > 1 and not rep_ok:
                        err = "incomplete_repeats_or_parse"
                    compares["edges_vs_sequential"] = {"error": err}

                cu_rc = rr_cu.returncode if rr_cu is not None else -1
                if (
                    rr_cu is None
                    or cu_rc != 0
                    or n_ok_cu != args.repeat
                    or not rows
                ):
                    binary_fail_msgs.append(
                        f"{key} cuda block={int(blk)} rc={cu_rc} tekrarlar={n_ok_cu}/{args.repeat}"
                    )

                payload = {
                    "cwd": str(cwd),
                    "cuda_block": int(blk),
                    "returncode": cu_rc,
                    "repeat": args.repeat,
                    "successful_repeats": n_ok_cu,
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
