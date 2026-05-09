# Scheduler — Entegrasyon, doğruluk ve GFLOPS/W analizi

Bu klasör **Emir Furkan Bazlı** rolündeki teslimleri kapsar: bileşenleri bir araya getirmek (sequential referansı, OpenMP, CUDA görüntü işleme), çıktıları karşılaştırmak, metrikleri toplamak ve GFLOPS / GFLOPS/W grafiklerini üretmek ve çalıştırma sürecini belgelemek.

## Görev özeti

| Görev | Bu repoda karşılığı |
| --- | --- |
| Zamanlayıcı mantığı ve entegrasyon | `run_benchmark.py` — ikilileri aynı girdi PGM ile çalıştırır; dizin yapısı ve çıktılar standartdır. |
| CPU + GPU çıktı karşılaştırması | `compare_pgms.py` ve `benchmark_tools.compare_pgm_files` — PGM piksel fark özeti ve tolerans kontrolü. |
| GFLOPS/W analizi ve grafikler | Stdout → `metrics.csv` ayrıştırması (`benchmark_tools`) + `plot_metrics.py` ile PNG grafikleri. |
| Dokümantasyon / rapora girdi | Bu dosya + `runs/<...>/summary.json` ve `runs/<...>/stdout.txt` gibi ham kayıtlar. |

CUDA tarafında stdout ve dosya adları için ekip arkadaşların verdiği sözdizimine uy; referans çıktılar **Sequential** üretimi kabul edilir.

## Dal ve Git notları

Çalışmalar **`scheduler`** dalında yapılmalı; **`main` dalına doğrudan commit/push etmeyin** — birleştirmeyi sorumlu yapacak.

Her gün: GitHub Desktop’ta **Fetch origin** → gerekiyorsa **Pull**.

## Ön koşullar

- Üç ikiliyi de (veya sırayla derleyerek) oluşturun:
  - `sequential` dalı → `image_processing`
  - `openmp` dalı → `image_processing_omp`
  - `cuda` dalı → `image_processing_cuda`
- PGM girdiler: `images/512x512.pgm`, `images/1024x1024.pgm`, `images/7680x4320.pgm`
- Python 3.10+ önerilir. Grafik için:

```bash
cd scheduler
pip install -r requirements.txt
```

Ölçüm notları:

- Sequential / OpenMP enerji için `/sys/class/powercap/...` (Intel RAPL) okunması bazen **`sudo`** veya uygun izin gerektirir (yoksa `Energy: N/A` normaldir).
- CUDA enerji yaklaşımı `nvidia-smi` gücü × çekirdek süresi; Linux + NVIDIA ortamında test edilir.

## Toplu doğruluk ve zamanlama: `run_benchmark.py`

Çalıştırma klasöründe betik klasörüdür (`scheduler`). Örnek (ikili yollarını kendi build dizininize göre değiştirin):

```bash
cd scheduler

python run_benchmark.py \
  --seq-bin ../build/seq/image_processing \
  --omp-bin ../build/omp/image_processing_omp \
  --cuda-bin ../build/cuda/image_processing_cuda \
  --omp-threads 8 \
  --cuda-blocks 8 16 32
```

Sadece belirli görseller için:

```bash
python run_benchmark.py \
  --seq-bin ../build/seq/image_processing \
  --omp-bin ../build/omp/image_processing_omp \
  --cuda-bin ../build/cuda/image_processing_cuda \
  --image ../images/512x512.pgm \
  --image ../images/1024x1024.pgm
```

### Üretilen yapı

`scheduler/runs/<run_adı>/` altında özetle:

| Yol | Açıklama |
| --- | --- |
| `sequential/<stem>/stdout.txt`, `stderr.txt` | Sequential metrik çıktısı |
| `openmp/t<Nt>/<stem>/…` | OpenMP (`Nt` iş parçacığı) |
| `cuda/b<B>/<stem>/…` | CUDA (`B×B` blok) |
| `metrics.csv` | Tüm fazlar için zaman / enerji / GFLOPS / GFLOPS/W satırları |
| `summary.json` | PGM kıyasları (`comparison`) ve çıkış kodları |

Çıkış kodları:

- `0`: ikililer düzgün, PGM kıyasları tolerans içinde ve eksik dosya yok.
- `1`: programlar çalışmış olabilir; en az bir kıyas tolerans üstü veya çıktı eksik.
- `2`: Sequential / OpenMP / CUDA süreçlerinden biri hata koduyla dönmüş (`stderr.txt` bakın).

Piksel toleransı için `--compare-threshold` (varsayılan `3`; ` uchar` çıktılar için uygun).

### İki PGM’yi tek başına karşılaştırma

```bash
python compare_pgms.py path/to/a.pgm path/to/b.pgm --threshold 3
```

Uyumlu ise çıkış `0`; aksi halde `1`.

## Grafik üretimi: `plot_metrics.py`

`run_benchmark` sonrası:

```bash
python plot_metrics.py --csv runs/<run_adı>/metrics.csv --output-dir graphs
```

Her çözünürlük ve faz için `gflops_*` ile `gflops_w_*` başlıklı PNG’ler oluşturur (Örn: `gflops_w_gaussian_512x512.png`).

`GFLOPS/W` CSV’de boş ise (Energy veya güç ölçülemediyse) ilgili çubuk atlanır.

## Rapor için önerilen test listesi

1. Sequential referans doğrulaması — referans PGM’leri `sequential/` çalıştırmasından üretin.
2. OpenMP doğruluğu — aynı girdi PGM; `runs/.../openmp/.../comparison` ile diff özetleri.
3. CUDA doğruluğu — bloklar **8, 16, 32** için aynı kıyas.
4. Performans karşılığı — `metrics.csv` + grafikleri rapora yapıştırın; blok ve iş parçacığı etiketi başlıkta belirtilsin.
5. Energy / GFLOPS/W — ölçülemeyen satırlar için “ölçüm koşulu” notu yazın (`N/A`, yetkisiz RAPL vb.).

Bu akış eksiksiz ise entegrasyon, doğruluk ve grafik üretimi ekip gereksinimleriyle uyumludur; matris çarpımı gibi başka bileşenler eklendiğinde aynı desen (`stdout` düzeni + PGM veya çıktı dosyası + CSV) yeniden kullanılabilir.
