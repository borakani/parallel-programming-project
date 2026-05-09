"""
Ortak yardımcılar: PGM okuma, piksel karşılaştırması, image pipeline stdout ayrıştırması.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

FLOAT_METRIC_KEYS = ("time_s", "energy_j", "gflops", "gflops_w")

__all__ = [
    "read_pgm_p5",
    "compare_pgm_files",
    "parse_image_pipeline_stdout",
    "average_parsed_metric_rows",
    "append_metrics_csv",
    "write_metrics_csv",
    "write_json",
    "METRIC_FIELDNAMES",
    "FLOAT_METRIC_KEYS",
]


def _readline_skip_comments(f: BinaryIO) -> bytes:
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Beklenmeyen dosya sonu (PGM başlığı)")
        s = line.decode("ascii", errors="replace").strip()
        if s.startswith("#"):
            continue
        if s:
            return line


def read_pgm_p5(path: Path) -> tuple[int, int, bytes]:
    """P5 (binary) PGM okur; yorum satırlarını atlar. Döndürür: (W, H, piksel baytları)."""
    with path.open("rb") as f:
        magic = _readline_skip_comments(f).decode("ascii", errors="replace").strip()
        if magic != "P5":
            raise ValueError(f"P5 destekleniyor, bulunan: {magic!r} ({path})")

        dim_line = _readline_skip_comments(f).decode("ascii", errors="replace").strip()
        parts = dim_line.split()
        if len(parts) < 2:
            raise ValueError(f"Geçersiz boyut satırı: {dim_line!r}")
        w, h = int(parts[0]), int(parts[1])

        maxval_line = _readline_skip_comments(f).decode("ascii", errors="replace").strip()
        maxv = int(maxval_line.split()[0])
        if maxv != 255:
            raise ValueError(f"Yalnızca maxval=255 destekleniyor, bulunan: {maxv}")

        payload = f.read()
    expected = w * h
    if len(payload) != expected:
        raise ValueError(
            f"Piksel boyutu uyuşmuyor: beklenen {expected}, gelen {len(payload)} ({path})"
        )
    return w, h, payload


@dataclass
class PgmCompareResult:
    path_a: str
    path_b: str
    width: int
    height: int
    max_abs_diff: int
    mean_abs_diff: float
    rmse: float
    differing_pixels: int
    threshold: int
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compare_pgm_files(
    path_a: Path,
    path_b: Path,
    *,
    threshold: int = 3,
) -> PgmCompareResult:
    wa, ha, ba = read_pgm_p5(path_a)
    wb, hb, bb = read_pgm_p5(path_b)
    if (wa, ha) != (wb, hb):
        raise ValueError(
            f"Boyutlar farklı: {path_a} {wa}x{ha} vs {path_b} {wb}x{hb}"
        )
    a = np.frombuffer(ba, dtype=np.uint8)
    b = np.frombuffer(bb, dtype=np.uint8)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    max_abs = int(d.max())
    mean_abs = float(d.mean())
    rmse = float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))
    differing = int(np.count_nonzero(d > threshold))

    waiv = differing == 0
    return PgmCompareResult(
        path_a=str(path_a),
        path_b=str(path_b),
        width=wa,
        height=ha,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rmse=rmse,
        differing_pixels=differing,
        threshold=threshold,
        ok=max_abs <= threshold,
    )


_SECTION_HDR = re.compile(
    r"^===\s*(Gaussian Blur|Sobel Edge Detection)\b(.*)===\s*$"
)
_IMG = re.compile(r"Image size:\s*(\d+)x(\d+)")
_TIME = re.compile(r"Time:\s*(\S+)\s+seconds")
_ENERGY = re.compile(r"Energy:\s*(\S+)\s+Joules")
_ENERGY_NA = re.compile(r"Energy:\s*N/A")
_GFLOPS = re.compile(r"GFLOPS:\s*(\S+)")
_GFLOPSW = re.compile(r"GFLOPS/W:\s*(\S+)")
_GFLOPSW_NA = re.compile(r"GFLOPS/W:\s*N/A")


def _parse_float_token(tok: str) -> float | None:
    tok = tok.strip()
    if tok in ("N/A", "nan", ""):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _infer_backend(section_line_tail: str) -> tuple[str | None, str | None, str]:
    tail = section_line_tail.strip()
    m_t = re.search(r"Threads:\s*(\d+)", tail)
    m_b = re.search(r"Block:\s*(\d+)", tail)
    if m_t and m_b:
        return ("unknown", None, tail)
    if m_t and not m_b:
        return ("openmp", m_t.group(1), f"threads={m_t.group(1)}")
    if m_b and not m_t:
        return ("cuda", m_b.group(1), f"block={m_b.group(1)}")
    return ("sequential", None, "sequential")


def parse_image_pipeline_stdout(text: str) -> list[dict[str, Any]]:
    """
    Sequential / OpenMP / CUDA uyumlu stdout'tan Gaussian ve Sobel metrik bloklarını çıkarır.
    """
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        backend: str,
        param_value: str | None,
        display_label: str,
        phase: str,
        w: int,
        h: int,
        t: float | None,
        ej: float | None,
        g: float | None,
        gw: float | None,
    ) -> None:
        threads = ""
        block = ""
        if backend == "openmp":
            threads = param_value or ""
        elif backend == "cuda":
            block = param_value or ""
        lbl = display_label or f"{phase}|{param_value or 'default'}"

        rows.append(
            {
                "phase": phase,
                "backend": backend or "unknown",
                "threads": threads,
                "block": block,
                "label": lbl,
                "width": str(w),
                "height": str(h),
                "time_s": "" if t is None else f"{t:.10g}",
                "energy_j": "" if ej is None else f"{ej:.10g}",
                "gflops": "" if g is None else f"{g:.10g}",
                "gflops_w": "" if gw is None else f"{gw:.10g}",
            }
        )

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m_hdr = _SECTION_HDR.match(stripped)
        if not m_hdr:
            i += 1
            continue

        name, tail = m_hdr.group(1), m_hdr.group(2)
        phase = "gaussian" if name.startswith("Gaussian") else "sobel"
        backend, param_value, display_label = _infer_backend(tail or "")

        i += 1
        w = h = None
        t = ej = g = gw = None
        while i < len(lines):
            ln = lines[i].rstrip()
            if ln.strip().startswith("==="):
                break
            mm = _IMG.search(ln)
            if mm:
                w, h = int(mm.group(1)), int(mm.group(2))
            mm = _TIME.search(ln)
            if mm:
                t = _parse_float_token(mm.group(1))
            if _ENERGY_NA.search(ln):
                ej = None
            else:
                mm = _ENERGY.search(ln)
                if mm:
                    ej = _parse_float_token(mm.group(1))
            mm = _GFLOPS.search(ln)
            if mm:
                g = _parse_float_token(mm.group(1))
            if _GFLOPSW_NA.search(ln):
                gw = None
            else:
                mm = _GFLOPSW.search(ln)
                if mm:
                    gw = _parse_float_token(mm.group(1))
            i += 1

        if w is not None and h is not None and t is not None:
            append_row(
                backend=backend or "unknown",
                param_value=param_value,
                display_label=display_label,
                phase=phase,
                w=w,
                h=h,
                t=t,
                ej=ej,
                g=g,
                gw=gw,
            )

    return rows


def average_parsed_metric_rows(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Ayni ikiliden N kez ayristirilmis metrik dosyalari (her biri Gaussian+Sobel satirlari).
    Sayisal kolonlari faz bakimindan ortalar; `samples` = N.

    Kosum faz sayisi uyusmuyorsa veya faz yapilandirmasi kosumlar arasinda farkliysa bos liste dondurur.
    """
    if not runs:
        return []
    n = len(runs)
    n_phases = len(runs[0])
    by_phase: dict[str, list[dict[str, Any]]] = {}

    for run in runs:
        if len(run) != n_phases:
            return []
        for row in run:
            by_phase.setdefault(row["phase"], []).append(row)

    phases_in_order = [r["phase"] for r in runs[0]]
    rows_out: list[dict[str, Any]] = []

    for ph in phases_in_order:
        grp = by_phase.get(ph)
        if not grp or len(grp) != n:
            return []

        merged = dict(grp[0])
        merged.pop("samples", None)
        identical_keys = {"phase", "backend", "threads", "block", "label", "width", "height"}
        for candidate in grp[1:]:
            for k in identical_keys:
                if merged.get(k) != candidate.get(k):
                    return []

        for key in FLOAT_METRIC_KEYS:
            nums = [
                fv
                for item in grp
                if (fv := _parse_float_token(str(item.get(key, "")))) is not None
            ]
            if nums:
                merged[key] = f"{statistics.mean(nums):.10g}"
            else:
                merged[key] = ""

        merged["samples"] = str(n)
        rows_out.append(merged)

    return rows_out


METRIC_FIELDNAMES = [
    "run_name",
    "image_stem",
    "phase",
    "backend",
    "threads",
    "block",
    "label",
    "width",
    "height",
    "time_s",
    "energy_j",
    "gflops",
    "gflops_w",
    "samples",
]


def append_metrics_csv(
    path: Path,
    run_name: str,
    image_stem: str,
    parsed_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=METRIC_FIELDNAMES)
        if new_file:
            w.writeheader()
        for r in parsed_rows:
            row = {k: "" for k in METRIC_FIELDNAMES}
            row.update(r)
            row.setdefault("samples", "")
            row["run_name"] = run_name
            row["image_stem"] = image_stem
            w.writerow(row)


def write_metrics_csv(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=METRIC_FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

