# -*- coding: utf-8 -*-
"""
13_colab_session.py
-------------------
AC-CA - Colab oturumunu sifirdan ya da baglanti kopmasindan sonra hazirlar.

NEDEN
  Colab koptugunda /content TAMAMEN silinir: veri, scriptler, sam2 paketi,
  SAM2 kontrol noktasi ve yerel egitim kontrol noktalari gider. Drive'daki
  acca_runs/ (metrics.csv, fold*_best.pt, fold*_last.pt, progress.json) KALIR.
  Bu script gerisini geri getirir. Her adim idempotent: yapilmissa atlanir,
  yani ayni oturumda tekrar cagirmak zararsizdir.

  1) Drive bagli mi
  2) veri paketi  : MyDrive/acca_colab_bundle.zip -> /content/acca
  3) scriptler    : MyDrive/acca_scripts/*.py -> /content/acca/scripts
                    DIKKAT: paketin icindeki 03/04 ESKI surumdur; bu adim
                    onlarin USTUNE yazar. Sira bu yuzden paket -> scriptler.
  4) surum kontrolu (03/04 guncel mi; eskiyse DURUR)
  5) sam2 paketi
  6) SAM2 kontrol noktasi (once Drive kopyasi, yoksa indirir ve Drive'a saklar)
  7) GPU
  8) kosu durumu  : biten foldlar, yarim foldun Drive'daki son kopyasi ve
                    tekrar kosulacak epoch sayisi

DEVAM MANTIGI (04_train.py)
  Yerel kontrol noktasi silindigi icin egitim Drive'daki fold{k}_last.pt'den
  devam eder. Bu dosya her SYNC_EVERY (5) epoch'ta yazildigi icin son
  kopyadan sonraki en fazla 4 epoch yeniden kosulur. metrics.csv'de bu
  epoch'lar iki kez yer alir; 12_lowres_prior.report() (fold, epoch)
  tekrarlarinda SONUNCUYU tutar.

KULLANIM (her yeni oturumun ILK hucresi)
    from google.colab import drive
    drive.mount("/content/drive")
    import importlib.util, sys
    def load(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s); sys.modules[name] = m
        s.loader.exec_module(m); return m
    S = load("S", "/content/drive/MyDrive/acca_scripts/13_colab_session.py")
    S.prepare(run="sam2_prior_lowres32")

  prepare() sonunda bir sonraki hucreyi kosunun durumuna gore yazdirir.
  Egitim ayarlarini elle girmek yerine S.trainer(run) kullanin: 04_train.py'yi
  yukler, RUN_CONFIGS'teki ayarlari uygular ve kosu adini DOGRULAR. Boylece
  kopma sonrasi PRIOR_GRID gibi bir ayari unutup yanlis klasore yazmak
  mumkun olmaz.
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import importlib
import importlib.util
import subprocess
import urllib.request

# ================================ CONFIG ==============================
DRIVE_ROOT    = "/content/drive/MyDrive"
BUNDLE_ZIP    = "/content/drive/MyDrive/acca_colab_bundle.zip"   # 02 numarali scriptin ciktisi
DRIVE_SCRIPTS = "/content/drive/MyDrive/acca_scripts"            # guncel .py dosyalari
DRIVE_CACHE   = "/content/drive/MyDrive/acca_cache"              # SAM2 kontrol noktasi kopyasi
RUNS_DIR      = "/content/drive/MyDrive/acca_runs"

BASE_DIR      = "/content/acca"
SCRIPTS_DIR   = "/content/acca/scripts"
CKPT_PATH     = "/content/sam2.1_hiera_base_plus.pt"
CKPT_URL      = ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
                 "sam2.1_hiera_base_plus.pt")
SAM2_PIP      = "git+https://github.com/facebookresearch/sam2.git"

N_FOLDS       = 5
EPOCHS        = 100
SYNC_EVERY    = 5            # 04_train.py ile AYNI olmali

# Dosya adi -> icinde bulunmasi gereken ifadeler (eski surumu yakalamak icin)
REQUIRED = {
    "temporal_pair_dataloader_v2.py": ["def build_loaders"],
    "03_model.py":        ["def degrade_prior", "prior_grid_tag"],
    "04_train.py":        ["PRIOR_GRID", "dl_mod=dl"],
    "12_lowres_prior.py": ["def report"],
    "14_binary_prior.py": ["def tau_sweep"],
}

# Kosu adi -> 04_train.py ayarlari. Yeni deney eklendikce buraya bir satir.
RUN_CONFIGS = {
    "sam2_prior_lowres32": dict(BACKBONE="sam2", PRIOR_MODE="channel",
                                USE_PRIOR=True, USE_STEM=True, PRIOR_GRID=32,
                                PRIOR_BINARY=False),
    "sam2_prior_binary":   dict(BACKBONE="sam2", PRIOR_MODE="channel",
                                USE_PRIOR=True, USE_STEM=True, PRIOR_GRID=None,
                                PRIOR_BINARY=True),
}
# ======================================================================


def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print(t); print("=" * 72)


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def _md5(path, n=8):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ------------------------------- adimlar ------------------------------
def step_drive():
    hr("1) DRIVE")
    if not os.path.isdir(DRIVE_ROOT):
        print("  [-] Drive bagli degil. Once:\n"
              "      from google.colab import drive; drive.mount('/content/drive')")
        return False
    print(f"  [+] {DRIVE_ROOT}")
    return True


def step_bundle():
    hr("2) VERI PAKETI")
    csv = os.path.join(BASE_DIR, "folds_pairs.csv")
    if os.path.exists(csv) and os.path.isdir(os.path.join(BASE_DIR, "_prior_cache")):
        print(f"  [atla] zaten acik: {BASE_DIR}")
    else:
        if not os.path.exists(BUNDLE_ZIP):
            print(f"  [-] paket yok: {BUNDLE_ZIP}\n"
                  f"      02_make_colab_bundle.py ciktisini Drive'a yukleyin.")
            return False
        local = os.path.join(os.path.dirname(BASE_DIR), os.path.basename(BUNDLE_ZIP))
        print(f"  Drive'dan kopyalaniyor ({os.path.getsize(BUNDLE_ZIP)/1e6:.0f} MB)...")
        shutil.copy2(BUNDLE_ZIP, local)          # Drive'dan dogrudan acmak yavas
        print("  aciliyor...")
        with zipfile.ZipFile(local) as z:
            z.extractall(BASE_DIR)
        os.remove(local)                         # /content diskini bosalt
        print(f"  [+] acildi: {BASE_DIR}")

    # butunluk: CSV'deki cift sayisi kadar prior PNG olmali
    import pandas as pd
    n = len(pd.read_csv(csv, encoding="utf-8-sig"))
    caches = [d for d in os.listdir(os.path.join(BASE_DIR, "_prior_cache"))
              if os.path.isdir(os.path.join(BASE_DIR, "_prior_cache", d))]
    ok = False
    for c in caches:
        k = len([f for f in os.listdir(os.path.join(BASE_DIR, "_prior_cache", c, "prior"))
                 if f.endswith(".png")])
        good = (k == n)
        ok |= good
        print(f"  {'[+]' if good else '[!]'} onbellek {c}: {k} prior / {n} cift")
    return ok


def step_scripts():
    hr("3) SCRIPTLER (Drive -> /content, paketteki eski surumlerin ustune)")
    if not os.path.isdir(DRIVE_SCRIPTS):
        print(f"  [-] klasor yok: {DRIVE_SCRIPTS}\n"
              f"      Guncel .py dosyalarini (en az {', '.join(REQUIRED)}) buraya koyun.")
        return False
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DRIVE_SCRIPTS) if f.endswith(".py"))
    if not files:
        print(f"  [-] {DRIVE_SCRIPTS} icinde .py yok")
        return False
    for f in files:
        src, dst = os.path.join(DRIVE_SCRIPTS, f), os.path.join(SCRIPTS_DIR, f)
        old = _md5(dst) if os.path.exists(dst) else None
        shutil.copy2(src, dst)
        new = _md5(dst)
        note = ("yeni" if old is None else
                "ayni" if old == new else f"paketteki {old} surumunun USTUNE yazildi")
        print(f"  {f:<34} md5 {new}   {note}")
    return True


def step_versions():
    hr("4) SURUM KONTROLU")
    ok = True
    for f, markers in REQUIRED.items():
        p = os.path.join(SCRIPTS_DIR, f)
        if not os.path.exists(p):
            print(f"  [-] {f}: YOK"); ok = False; continue
        with open(p, encoding="utf-8") as fh:
            txt = fh.read()
        miss = [m for m in markers if m not in txt]
        if miss:
            print(f"  [-] {f}: ESKI surum (eksik: {miss}) -> Drive'daki dosyayi guncelleyin")
            ok = False
        else:
            print(f"  [+] {f}")
    return ok


def step_sam2(install=True):
    hr("5) SAM2 PAKETI")
    try:
        import sam2
        print(f"  [atla] kurulu: {os.path.dirname(sam2.__file__)}")
        return True
    except ImportError:
        pass
    if not install:
        print("  [-] kurulu degil (install=False)")
        return False
    print("  kuruluyor (1-2 dk)...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", SAM2_PIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("  [-] kurulum basarisiz:\n" + r.stderr[-800:])
        return False
    importlib.invalidate_caches()
    try:
        import sam2
        print(f"  [+] kuruldu: {os.path.dirname(sam2.__file__)}")
        return True
    except ImportError:
        print("  [-] kuruldu ama ice aktarilamadi. Runtime > Restart session, "
              "sonra ilk hucreyi tekrar calistirin.")
        return False


def step_ckpt():
    hr("6) SAM2 KONTROL NOKTASI")
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1e8:
        print(f"  [atla] mevcut: {CKPT_PATH}")
        return True
    cached = os.path.join(DRIVE_CACHE, os.path.basename(CKPT_PATH))
    if os.path.exists(cached) and os.path.getsize(cached) > 1e8:
        print(f"  Drive kopyasindan aliniyor: {cached}")
        shutil.copy2(cached, CKPT_PATH)
    else:
        print(f"  indiriliyor: {CKPT_URL}")
        tmp = CKPT_PATH + ".tmp"
        urllib.request.urlretrieve(CKPT_URL, tmp)
        os.replace(tmp, CKPT_PATH)
        try:
            os.makedirs(DRIVE_CACHE, exist_ok=True)
            shutil.copy2(CKPT_PATH, cached)
            print(f"  sonraki oturumlar icin Drive'a kopyalandi: {cached}")
        except Exception as e:
            print(f"  [not] Drive'a kopyalanamadi ({type(e).__name__}); sorun degil")
    ok = os.path.getsize(CKPT_PATH) > 1e8
    print(f"  {'[+]' if ok else '[-]'} {CKPT_PATH} ({os.path.getsize(CKPT_PATH)/1e6:.0f} MB)")
    return ok


def step_gpu():
    hr("7) GPU")
    try:
        import torch
    except ImportError:
        print("  [-] torch yok"); return False
    if not torch.cuda.is_available():
        print("  [-] GPU yok: Runtime > Change runtime type > GPU")
        return False
    p = torch.cuda.get_device_properties(0)
    print(f"  [+] {p.name}  {p.total_memory/1024**3:.1f} GB")
    return True


def status(run):
    """Kosunun Drive'daki durumu. Egitim ve GPU gerektirmez."""
    import pandas as pd
    hr(f"8) KOSU DURUMU: {run}")
    rd = os.path.join(RUNS_DIR, run)
    st = dict(run=run, exists=os.path.isdir(rd), done=[], partial={}, started=False)
    if not st["exists"]:
        print("  henuz baslamamis (klasor yok)")
        return st
    prog = os.path.join(rd, "progress.json")
    if os.path.exists(prog):
        with open(prog, encoding="utf-8") as f:
            st["done"] = sorted(json.load(f).get("done_folds", []))
    m = None
    mcsv = os.path.join(rd, "metrics.csv")
    if os.path.exists(mcsv):
        m = pd.read_csv(mcsv)
        st["started"] = len(m) > 0

    for k in range(N_FOLDS):
        files = dict(best=os.path.exists(os.path.join(rd, f"fold{k}_best.pt")),
                     last=os.path.exists(os.path.join(rd, f"fold{k}_last.pt")),
                     pairs=os.path.exists(os.path.join(rd, f"fold{k}_val_pairs.csv")))
        if k in st["done"]:
            flag = "" if files["best"] and files["pairs"] else "   <<< best/val_pairs EKSIK"
            print(f"  fold {k}: BITTI{flag}")
            continue
        logged = None
        if m is not None and (m.fold == k).any():
            logged = int(m[m.fold == k].epoch.max())
        if logged is None:
            print(f"  fold {k}: baslamamis")
            continue
        synced = (logged + 1) // SYNC_EVERY * SYNC_EVERY - 1
        if files["last"] and synced >= 0:
            redo = logged - synced
            print(f"  fold {k}: YARIM - loglanan son epoch {logged}, Drive kopyasi ~epoch "
                  f"{synced} -> {synced + 1}. epoch'tan devam ({redo} epoch tekrar kosulur)")
        else:
            print(f"  fold {k}: YARIM - loglanan son epoch {logged}, Drive kopyasi YOK "
                  f"-> fold BASTAN kosulur")
            synced = -1
        st["partial"][k] = dict(logged=logged, resume_from=synced + 1)
    return st


# ------------------------------ ana giris -----------------------------
def prepare(run=None, install=True):
    """
    Oturumu hazirlar. Basarisiz adimda durur ve nedenini yazar.
    run verilirse kosunun durumunu gosterir ve siradaki hucreyi yazdirir.
    """
    steps = [("drive", step_drive), ("paket", step_bundle), ("scriptler", step_scripts),
             ("surum", step_versions), ("sam2", lambda: step_sam2(install)),
             ("ckpt", step_ckpt), ("gpu", step_gpu)]
    for name, fn in steps:
        if not fn():
            hr("DURDU")
            print(f"  '{name}' adimi basarisiz. Yukaridaki mesaja gore duzeltip "
                  f"S.prepare(...) hucresini tekrar calistirin; biten adimlar atlanir.")
            return False

    if run is None:
        hr("HAZIR"); print("  Oturum hazir."); return True

    st = status(run)
    hr("SIRADAKI HUCRE")
    known = run in RUN_CONFIGS
    head = (f'T = S.trainer("{run}")' if known else
            f'T = S.load("T", "{SCRIPTS_DIR}/04_train.py")\n'
            f'# RUN_CONFIGS icinde "{run}" yok: ayarlari ONCEKI oturumla ayni girin\n'
            f'# ve assert T.run_tag() == "{run}" ile dogrulayin')
    if not st["started"]:
        print(f"""  Kosu baslamamis. Once dogrulama ve duman testi:

    L = S.load("L", "{SCRIPTS_DIR}/12_lowres_prior.py")
    L.info_loss()          # daha once kaydettiyseniz atlayin
    L.verify()
    {head.replace(chr(10), chr(10) + '    ')}
    T.smoke_test(n=10, epochs=30)
    T.run_all()""")
    elif len(st["done"]) == N_FOLDS:
        print(f"""  Tum foldlar bitmis. Rapor:

    L = S.load("L", "{SCRIPTS_DIR}/12_lowres_prior.py")
    L.report()""")
    else:
        print(f"""  Kosu yarim. Dogrulama ve duman testi daha once gectiyse DOGRUDAN devam:

    {head.replace(chr(10), chr(10) + '    ')}
    T.run_all()            # biten foldlari atlar, yarim foldu Drive kopyasindan surdurur""")
    return True


def trainer(run):
    """
    04_train.py'yi yukler, RUN_CONFIGS[run] ayarlarini uygular ve kosu adini
    dogrular. Kopma sonrasi ayar unutulup yanlis klasore yazilmasini engeller.
    """
    if run not in RUN_CONFIGS:
        raise KeyError(f"RUN_CONFIGS icinde '{run}' yok. Ayarlari 13_colab_session.py'ye ekleyin.")
    T = load("T", os.path.join(SCRIPTS_DIR, "04_train.py"))
    for k, v in RUN_CONFIGS[run].items():
        if not hasattr(T, k):
            raise AttributeError(f"04_train.py'de {k} yok - eski surum olabilir.")
        setattr(T, k, v)
    T.SAM2_CKPT = CKPT_PATH
    T.RUN_NAME = None
    tag = T.run_tag()
    if tag != run:
        raise RuntimeError(f"ayarlar '{tag}' kosusunu uretiyor, beklenen '{run}'.")
    if T.SYNC_EVERY != SYNC_EVERY:
        print(f"  [not] 04_train.SYNC_EVERY={T.SYNC_EVERY}, bu scriptte {SYNC_EVERY}: "
              f"durum raporundaki devam epoch'u yaklasik olur.")
    print(f"[trainer] {run}: " + ", ".join(f"{k}={v}" for k, v in RUN_CONFIGS[run].items()))
    return T


if __name__ == "__main__":
    prepare()
