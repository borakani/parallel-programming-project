# Scheduler (C) — entegrasyon ve kıyas

Python sürümü kaldırıldı; tüm araçlar **C99/POSIX** ile burada derlenir (**Linux / WSL** önerilir). `run_benchmark` **fork/exec** kullanır; yalnızca Windows’ta `run_benchmark.exe` oluşturmak için değil, POSIX ortamda derlenmelidir.

## Derleme

```bash
cd scheduler
make clean && make all
```

Üretilen ikililer (aynı klasörde):

- **`compare_pgms`** — iki PGM (P5) piksel kıyası  
- **`decide_scheduler`** — Checkpoint güç bütçesi seçimi (`sche_policy`)  
- **`run_benchmark`** — sırayla sequential / OpenMP / CUDA görüntü işleme ve `metrics.csv`

## Görev özeti

| Görev | C karşılığı |
| --- | --- |
| Zamanlayıcı ve entegrasyon | `run_benchmark.c` + `sche_execposix.c` |
| Checkpoint güç bütçesi | `sche_policy.c` + `decide_scheduler.c` |
| PGM kıyas | `sche_pgm.c` + `compare_pgms.c` |
| Metrik çıktısı | `sche_metric_parse.c`, `sche_metric_avg.c`, `sche_csv.c` |
| Grafikler | Bu sürümde otomatik PNG yok — `metrics.csv`’yi tablo/grafik aracına aktarın. |

## `run_benchmark`

**Ortak:** `--repo` varsayılan `.` (`./images/*.pgm` aranır), `--output-dir` varsayılan `scheduler/runs`, `--run-name` boşsa UTC zaman damgası, `--repeat` varsayılan `1`, `--compare-threshold` varsayılan `3`.

```bash
./run_benchmark \
  --seq-bin /tam/yol/image_processing \
  --omp-bin /tam/yol/image_processing_omp \
  --cuda-bin /tam/yol/image_processing_cuda \
  --omp-threads 8 \
  --cuda-blocks 8,16,32 \
  --repeat 5
```

Liste virgülle (`--omp-threads 1,2,4,8`). Özel PGM: `--image yol.pgm` (birden çok `--image` kullanılabilir).

Çıkış klasörü: `--output-dir` / `--run-name` / `{sequential,openmp,cuda}/…` yapısı Python sürümüyle uyumludur. Her tekrar `stdout_KK.txt`; `metrics.csv` içinde `samples` ve ortalanmış süreler/enerji.

Çıkış kodları yaklaşık: `0` tamam; `1` PGM kıyas tolerans üstü; `2` ikili hatası veya eksik PGM.

## `compare_pgms`

```bash
./compare_pgms -t 3 ref.pgm tst.pgm
```

Uyum için çıkış `0`; değilse `1`; argüman hatası `2`.

## `decide_scheduler`

```bash
./decide_scheduler --budget 150 --cpu-energy 12 --gpu-energy 9 --gpu-power 120

./decide_scheduler --budget 100 --cpu-energy 8 --gpu-energy 5 --poll-gpu

```

İsteğe bağlı `--exit-code` (`GPU` → 0).

## Ölçüm notları

- RAPL / `nvidia-smi` için izin ve ortam aynı; enerji satırı bazen `N/A`.

## Dal

Değişiklikler **`scheduler`** dalına; **`main`**’e doğrudan push etmeyin (takım politikası).

