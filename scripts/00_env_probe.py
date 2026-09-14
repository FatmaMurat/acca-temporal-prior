# -*- coding: utf-8 -*-
"""
02_make_colab_bundle.py
-----------------------
AC-CA calismasini Colab'a tasimak icin tek zip hazirlar.

Once DOGRULAR, sonra paketler. folds_pairs.csv'nin isaret ettigi her dosyayi
ve prior onbellegindeki her PNG'yi kontrol eder; eksik varsa paketlemeden
once soyler. Eksigi Colab'da kesfetmek yerine burada kesfetmek icin.

Icerik:
  images/            csv'de gecen tum kesitler (prev + curr)
  masks/             ayni
  folds_pairs.csv
  _prior_cache/<id>/ prior + pbin + pair_stats.csv + manifest.json
  scripts/           01, 02, dataloader v2

prev dosyalari da dahil edilir: boylece Colab'da onbellegi farkli TAU ile
yeniden uretebilirsiniz (orada dakikalar surer).

Turkce yol notu: zipfile unicode yollarla sorunsuz calisir, ancak arsiv ici
adlar ASCII goreli yollara cevrilir.
"""

import os
import sys
import json
import time
import hashlib
import zipfile

import pandas as pd

# =============================== CONFIG ===============================
BASE_DIR   = r"C:\Users\Fatma Murat\OneDrive\Masaüstü\AC-CA Mask\Dataset_anonim"
CSV_PATH   = os.path.join(BASE_DIR, "folds_pairs.csv")
CACHE_ROOT = os.path.join(BASE_DIR, "_prior_cache")

# scriptlerin bulundugu klasor (modelleme klasoru)
SCRIPT_DIR = r"C:\Users\Fatma Murat\OneDrive\Masaüstü\AC-CA Mask\modelleme"
SCRIPTS    = ["01_prior_cache_and_baseline.py",
              "temporal_pair_dataloader_v2.py",
              "02_make_colab_bundle.py",
              "03_model.py",
              "04_train.py"]

OUT_ZIP    = os.path.join(os.path.dirname(BASE_DIR), "acca_colab_bundle.zip")

# onbellek kimligi (01 numarali scriptin CONFIG'i ile ayni)
PRIOR_MODE, TAU_MM, DILATE_MM, SIGMA_MM = "distance", 15.0, 10.0, 10.0
DFOV_SKIP, ALIGN_BODY, IMG_SIZE = 1.10, True, 512

INCLUDE_PREV = True      # prev goruntu/maskeleri de al (onbellek yeniden uretimi icin)
# =====================================================================


def config_id():
    cfg = dict(mode=PRIOR_MODE, tau=TAU_MM, dil=DILATE_MM, sig=SIGMA_MM,
               skip=DFOV_SKIP, align=ALIGN_BODY, size=IMG_SIZE)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    tag = (f"{PRIOR_MODE}_tau{TAU_MM:g}" if PRIOR_MODE == "distance"
           else f"blur_d{DILATE_MM:g}_s{SIGMA_MM:g}")
    return f"{tag}_align{int(ALIGN_BODY)}_sz{IMG_SIZE}_{h}"


def collect():
    """Paketlenecek (mutlak_yol, arsiv_ici_yol) listesini kurar ve dogrular."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["pair_id"] = (df["patient_key"].astype(str) + "__"
                     + df["t_prev"].astype(str) + "_" + df["t_curr"].astype(str))

    items, missing = [], []

    def add(abs_p, arc):
        if os.path.exists(abs_p):
            items.append((abs_p, arc))
        else:
            missing.append(abs_p)

    # --- goruntu ve maskeler ---
    cols = ["curr_img", "curr_mask"] + (["prev_img", "prev_mask"] if INCLUDE_PREV else [])
    rels = set()
    for c in cols:
        rels |= set(df[c].astype(str))
    for rel in sorted(rels):
        add(os.path.join(BASE_DIR, rel), rel.replace("\\", "/"))

    # --- csv ---
    add(CSV_PATH, "folds_pairs.csv")

    # --- prior onbellegi ---
    cid = config_id()
    cdir = os.path.join(CACHE_ROOT, cid)
    if not os.path.isdir(cdir):
        missing.append(cdir + "  (ONBELLEK KLASORU)")
    else:
        for f in ("manifest.json", "pair_stats.csv"):
            add(os.path.join(cdir, f), f"_prior_cache/{cid}/{f}")
        for sub in ("prior", "pbin"):
            sd = os.path.join(cdir, sub)
            n = 0
            for pid in df["pair_id"]:
                add(os.path.join(sd, pid + ".png"), f"_prior_cache/{cid}/{sub}/{pid}.png")
                n += 1

    # --- scriptler ---
    for s in SCRIPTS:
        p = os.path.join(SCRIPT_DIR, s)
        if os.path.exists(p):
            items.append((p, f"scripts/{s}"))
        else:
            print(f"  [not] script bulunamadi, atlaniyor: {s}")

    return df, items, missing, cid


def main():
    print("=" * 66)
    print("1) DOGRULAMA")
    print("=" * 66)
    if not os.path.exists(CSV_PATH):
        print(f"  !! folds_pairs.csv yok: {CSV_PATH}")
        sys.exit(1)

    df, items, missing, cid = collect()
    print(f"  cift sayisi        : {len(df)}")
    print(f"  onbellek kimligi   : {cid}")
    print(f"  paketlenecek dosya : {len(items)}")

    if missing:
        print(f"\n  !! EKSIK {len(missing)} DOSYA - paketleme durduruldu:")
        for m in missing[:20]:
            print(f"     {m}")
        if len(missing) > 20:
            print(f"     ... ve {len(missing)-20} tane daha")
        print("\n  Onbellek eksikse: 01_prior_cache_and_baseline.py'yi calistirin.")
        print("  Goruntu/maske eksikse: BASE_DIR yanlis olabilir.")
        sys.exit(1)
    print("  [OK] her dosya yerinde")

    total = sum(os.path.getsize(p) for p, _ in items)
    print(f"  ham boyut          : {total/1e6:.1f} MB")

    print("\n" + "=" * 66)
    print("2) PAKETLEME")
    print("=" * 66)
    t0 = time.time()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, (p, arc) in enumerate(items):
            z.write(p, arc)
            if (i + 1) % 250 == 0:
                print(f"    {i+1}/{len(items)} ...")
    zs = os.path.getsize(OUT_ZIP)
    print(f"  [OK] {OUT_ZIP}")
    print(f"  zip boyutu         : {zs/1e6:.1f} MB  ({time.time()-t0:.1f} sn)")

    print("\n" + "=" * 66)
    print("3) COLAB TARAFI")
    print("=" * 66)
    print(f"""  a) Bu zip'i Google Drive'a yukleyin (MyDrive koklerine).
  b) Colab'da Runtime > Change runtime type > GPU secin.
  c) Ilk hucre:

     from google.colab import drive
     drive.mount('/content/drive')
     !cp /content/drive/MyDrive/{os.path.basename(OUT_ZIP)} /content/
     !cd /content && unzip -q {os.path.basename(OUT_ZIP)} -d acca
     !nvidia-smi --query-gpu=name,memory.total --format=csv

  ONEMLI: veriyi Drive'dan DEGIL /content/acca'dan okuyun. Drive uzerinden
  ornek ornek okumak egitimi I/O'ya baglar.

  Dataloader'da degisecek tek sey yollar:
     BASE_DIR   = "/content/acca"
     CACHE_ROOT = "/content/acca/_prior_cache"
  Turkce karakter sorunu Linux'ta zaten yok, np.fromfile sarmalayicilari
  oldugu gibi calisir.""")


if __name__ == "__main__":
    main()