# -*- coding: utf-8 -*-
"""
temporal_pair_dataloader_v2.py
------------------------------
AC-CA calismasi, modelleme asamasi - Adim 2.

v1'e gore farklar:
  1) PRIOR ONBELLEKTEN OKUNUR. 01_prior_cache_and_baseline.py'nin urettigi
     PNG'ler kullanilir. Epoch basina distanceTransform + 2x Otsu YOK.
  2) AUGMENTASYON. Geometrik donusumler image+prior+mask'e ORTAK, yogunluk
     donusumleri yalnizca image'a, prior'a ozel bozma (jitter / tau / dropout).
  3) SIZINTI ASSERT'I true_patient_id uzerinden.
  4) attach_mmpx kaldirildi. Tum meta pair_stats.csv'den gelir; v1'deki
     "erken return -> chrono_flag hic olusmuyor" ve "key_of etiket anahtarini
     dosya anahtarindan once deniyor" hatalari boylece ortadan kalkti.
  5) TOHUMLAMA. 5 fold x 2 omurga karsilastirmasi tekrarlanabilir.
  6) PRELOAD. 380 kesit + 244 prior RAM'e alinir (~165 MB). NUM_WORKERS=0
     zorunlulugu altinda en buyuk hizlanma buradan gelir.

Cikti (v1 ile ayni sozlesme):
    image : (3, H, W) float32 [0,1]   <- HAM. Normalizasyon model sarmalayicisinda.
    prior : (1, H, W) float32 [0,1]
    mask  : (1, H, W) float32 {0,1}

Turkce yol notu: cv2.imread/imwrite YOK. np.fromfile + cv2.imdecode.
"""

import os
import json
import random
import hashlib
import warnings

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# =============================== CONFIG ===============================
BASE_DIR   = r"C:\Users\Fatma Murat\OneDrive\Masaüstü\AC-CA Mask\Dataset_anonim"
CSV_PATH   = os.path.join(BASE_DIR, "folds_pairs.csv")
CACHE_ROOT = os.path.join(BASE_DIR, "_prior_cache")

IMG_SIZE       = 512
IMAGE_CHANNELS = 3
BATCH_SIZE     = 4
NUM_WORKERS    = 0          # Windows/Spyder
SEED           = 42
PRELOAD        = True       # kesitleri ve prior'lari RAM'e al

# --- onbellek kimligi: 01 numarali scriptin CONFIG'i ile AYNI olmali ---
PRIOR_MODE = "distance"
TAU_MM     = 15.0
DILATE_MM  = 10.0
SIGMA_MM   = 10.0
DFOV_SKIP  = 1.10
ALIGN_BODY = True

# ------------------------- AUGMENTASYON AYARI -------------------------
AUG = dict(
    # ortak geometrik (image + prior + mask ayni matrisle)
    hflip_p     = 0.5,      # aksiyel BT'de lateraliteyi bozar; not: asagiya bkz.
    rot_deg     = 10.0,
    scale_rng   = (0.90, 1.10),
    trans_frac  = 0.05,
    affine_p    = 0.8,

    # yalnizca goruntu
    bright_rng  = (-0.05, 0.05),
    contrast_rng= (0.90, 1.10),
    gamma_rng   = (0.85, 1.15),
    noise_sigma = 0.02,
    intensity_p = 0.8,

    # yalnizca prior
    prior_shift_mm = 8.0,   # kalinti kaymanin (medyan 14.1 mm) yarisi kadar jitter
    prior_gamma_rng= (0.70, 1.40),  # prior^g == tau'yu 15 -> [21.4, 10.7] mm oynatir
    prior_drop_p   = 0.10,  # prior'i tamamen sifirla
)
# hflip notu: 27 ciftte (%11) lezyon t-1 ayak izinin tamamen disinda; veri
# cesitliligi bu yuzden degerli. Ancak aksiyel BT'de yatay cevirme kalbi ters
# tarafa koyar. Omurga onceden egitilmis oldugu icin bu anatomik olarak
# imkansiz girdi zararli olabilir. Ablasyonda hflip_p=0.0 ile de olcun.
# =====================================================================


# ----------------------------- configure ------------------------------
def configure(base_dir=None, csv_path=None, cache_root=None,
              img_size=None, batch_size=None, num_workers=None,
              preload=None, seed=None):
    """
    Yollari ve ayarlari CALISMA ANINDA degistirir.

    Modul degiskenine dogrudan atama YETMEZ: fonksiyonlarin varsayilan
    argumanlari tanimlandiklari anda sabitlenir. Bu yuzden tum fonksiyonlar
    varsayilan olarak None alir ve degeri cagri aninda buradan okur.
    Colab'a gecerken cagrilmasi ZORUNLU.
    """
    global BASE_DIR, CSV_PATH, CACHE_ROOT, IMG_SIZE, BATCH_SIZE
    global NUM_WORKERS, PRELOAD, SEED
    if base_dir is not None:
        BASE_DIR = base_dir
        CSV_PATH = os.path.join(base_dir, "folds_pairs.csv")
        CACHE_ROOT = os.path.join(base_dir, "_prior_cache")
    if csv_path   is not None: CSV_PATH    = csv_path
    if cache_root is not None: CACHE_ROOT  = cache_root
    if img_size   is not None: IMG_SIZE    = img_size
    if batch_size is not None: BATCH_SIZE  = batch_size
    if num_workers is not None: NUM_WORKERS = num_workers
    if preload    is not None: PRELOAD     = preload
    if seed       is not None: SEED        = seed
    print(f"[dataloader] BASE_DIR   = {BASE_DIR}")
    print(f"[dataloader] CSV_PATH   = {CSV_PATH}")
    print(f"[dataloader] CACHE_ROOT = {CACHE_ROOT}")
    return dict(BASE_DIR=BASE_DIR, CSV_PATH=CSV_PATH, CACHE_ROOT=CACHE_ROOT)


# ------------------------------ Tohumlama ----------------------------
def set_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)


# ------------------------------ Okuyucular ---------------------------
def robust_read_gray(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Goruntu okunamadi: {path}")
    return img


def robust_read_mask(path):
    buf = np.fromfile(path, dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"Maske okunamadi: {path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


# --------------------------- Onbellek cozumu -------------------------
def config_id():
    cfg = dict(mode=PRIOR_MODE, tau=TAU_MM, dil=DILATE_MM, sig=SIGMA_MM,
               skip=DFOV_SKIP, align=ALIGN_BODY, size=IMG_SIZE)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    tag = (f"{PRIOR_MODE}_tau{TAU_MM:g}" if PRIOR_MODE == "distance"
           else f"blur_d{DILATE_MM:g}_s{SIGMA_MM:g}")
    return f"{tag}_align{int(ALIGN_BODY)}_sz{IMG_SIZE}_{h}"


def resolve_cache(cache_root=None, n_expected=None):
    cache_root = CACHE_ROOT if cache_root is None else cache_root
    """Onbellegi bulur ve manifest'i dogrular. Bayat onbellek sessizce gecmez."""
    cid = config_id()
    cdir = os.path.join(cache_root, cid)
    man = os.path.join(cdir, "manifest.json")
    if not os.path.exists(man):
        raise FileNotFoundError(
            f"Onbellek yok: {cdir}\n"
            f"Once 01_prior_cache_and_baseline.py'yi AYNI CONFIG ile calistirin "
            f"(mode={PRIOR_MODE}, tau={TAU_MM}, align={ALIGN_BODY}, size={IMG_SIZE}).")
    with open(man, encoding="utf-8") as f:
        meta = json.load(f)
    if n_expected is not None and meta.get("n_pairs") != n_expected:
        raise RuntimeError(
            f"Onbellek {meta.get('n_pairs')} cift icin uretilmis, CSV'de "
            f"{n_expected} cift var. Onbellegi OVERWRITE=True ile yenileyin.")
    return cdir, meta


# --------------------------- Augmentasyon ----------------------------
def _joint_affine(img, prior, mask, rng, a=AUG):
    """image+prior+mask'e AYNI matrisi uygular."""
    H, W = mask.shape
    if rng.random() < a["hflip_p"]:
        img, prior, mask = img[:, ::-1], prior[:, ::-1], mask[:, ::-1]
        img, prior, mask = (np.ascontiguousarray(x) for x in (img, prior, mask))

    if rng.random() < a["affine_p"]:
        ang = rng.uniform(-a["rot_deg"], a["rot_deg"])
        sc  = rng.uniform(*a["scale_rng"])
        tx  = rng.uniform(-a["trans_frac"], a["trans_frac"]) * W
        ty  = rng.uniform(-a["trans_frac"], a["trans_frac"]) * H
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        img   = cv2.warpAffine(img,   M, (W, H), flags=cv2.INTER_LINEAR,  borderValue=0)
        prior = cv2.warpAffine(prior, M, (W, H), flags=cv2.INTER_LINEAR,  borderValue=0)
        mask  = cv2.warpAffine(mask,  M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, prior, mask


def _intensity(img, rng, a=AUG):
    """Yalnizca goruntu. img: float32 [0,1]."""
    if rng.random() >= a["intensity_p"]:
        return img
    img = img * rng.uniform(*a["contrast_rng"]) + rng.uniform(*a["bright_rng"])
    img = np.clip(img, 0, 1)
    g = rng.uniform(*a["gamma_rng"])
    if abs(g - 1.0) > 1e-3:
        img = np.power(img, g, dtype=np.float32)
    if a["noise_sigma"] > 0:
        img = img + rng.normal(0, rng.uniform(0, a["noise_sigma"]), img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def _perturb_prior(prior, mm_per_px, rng, a=AUG):
    """
    Yalnizca prior. Modelin prior'in TAM geometrisine guvenmesini engeller.
      - dropout : prior'i sifirla (27/244 ciftte prior zaten yaniltici)
      - jitter  : mm cinsinden rastgele oteleme
      - gamma   : prior^g, distance modda tau'yu olceklemeye denk
    """
    if rng.random() < a["prior_drop_p"]:
        return np.zeros_like(prior)

    if a["prior_shift_mm"] > 0 and mm_per_px > 0:
        s_px = a["prior_shift_mm"] / float(mm_per_px)
        dx, dy = rng.normal(0, s_px), rng.normal(0, s_px)
        H, W = prior.shape
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        prior = cv2.warpAffine(prior, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)

    g = rng.uniform(*a["prior_gamma_rng"])
    if abs(g - 1.0) > 1e-3:
        prior = np.power(np.clip(prior, 0, 1), g, dtype=np.float32)
    return prior.astype(np.float32)


# ------------------------------ Dataset ------------------------------
class TemporalPairDataset(Dataset):
    def __init__(self, df, base_dir=None, cache_dir=None,
                 img_size=None, image_channels=None,
                 augment=False, use_prev_img=False, preload=None, seed=None):
        base_dir       = BASE_DIR       if base_dir is None else base_dir
        img_size       = IMG_SIZE       if img_size is None else img_size
        image_channels = IMAGE_CHANNELS if image_channels is None else image_channels
        preload        = PRELOAD        if preload is None else preload
        seed           = SEED           if seed is None else seed
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.image_channels = image_channels
        self.augment = augment
        self.use_prev_img = use_prev_img
        self._rng = np.random.default_rng(seed)
        self._img_cache, self._prior_cache = {}, {}
        if preload:
            self._preload()

    def __len__(self):
        return len(self.df)

    # ---- I/O ----
    def _load_img(self, rel):
        if rel in self._img_cache:
            return self._img_cache[rel]
        a = robust_read_gray(os.path.join(self.base_dir, str(rel)))
        if a.shape != (self.img_size, self.img_size):
            a = cv2.resize(a, (self.img_size,) * 2, interpolation=cv2.INTER_AREA)
        if self._img_cache is not None:
            self._img_cache[rel] = a
        return a

    def _load_mask(self, rel):
        a = robust_read_mask(os.path.join(self.base_dir, str(rel)))
        if a.shape != (self.img_size, self.img_size):
            a = cv2.resize(a, (self.img_size,) * 2, interpolation=cv2.INTER_NEAREST)
        return a

    def _load_prior(self, pair_id):
        if pair_id in self._prior_cache:
            return self._prior_cache[pair_id]
        p = os.path.join(self.cache_dir, "prior", pair_id + ".png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Onbellekte prior yok: {p}")
        a = robust_read_gray(p).astype(np.float32) / 255.0
        self._prior_cache[pair_id] = a
        return a

    def _preload(self):
        rels = set(self.df["curr_img"])
        if self.use_prev_img:
            rels |= set(self.df["prev_img"])
        for r in rels:
            self._load_img(r)
        for pid in self.df["pair_id"]:
            self._load_prior(pid)

    # ---- ornek ----
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        pid = r["pair_id"]

        img   = self._load_img(r["curr_img"]).astype(np.float32) / 255.0
        prior = self._load_prior(pid).copy()
        mask  = self._load_mask(r["curr_mask"]).astype(np.float32)

        if self.augment:
            rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))
            img, prior, mask = _joint_affine(img, prior, mask, rng)
            prior = _perturb_prior(prior, float(r["mm_per_px_512"]), rng)
            img = _intensity(img, rng)

        if self.image_channels == 3:
            image = np.repeat(img[None], 3, axis=0)
        else:
            image = img[None]

        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "prior": torch.from_numpy(np.ascontiguousarray(prior[None], dtype=np.float32)),
            "mask":  torch.from_numpy(np.ascontiguousarray((mask > 0.5)[None], dtype=np.float32)),
            "pair_id": pid,
            "patient_key": r["patient_key"],
            "fold": int(r["fold"]),
            "dice_copy": float(r["dice_copy_aligned"]),   # cift-bazli taban cizgisi
        }
        if self.use_prev_img:
            pim = self._load_img(r["prev_img"]).astype(np.float32) / 255.0
            pim = np.repeat(pim[None], 3, 0) if self.image_channels == 3 else pim[None]
            sample["prev_image"] = torch.from_numpy(np.ascontiguousarray(pim, dtype=np.float32))
        return sample


# --------------------------- Loader kurucu ---------------------------
def load_table(csv_path=None, cache_root=None):
    """folds_pairs.csv + pair_stats.csv birlestirir, mm/px@IMG_SIZE turetir."""
    csv_path   = CSV_PATH   if csv_path is None else csv_path
    cache_root = CACHE_ROOT if cache_root is None else cache_root
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["pair_id"] = (df["patient_key"].astype(str) + "__"
                     + df["t_prev"].astype(str) + "_" + df["t_curr"].astype(str))
    cdir, meta = resolve_cache(cache_root, n_expected=len(df))
    st = pd.read_csv(os.path.join(cdir, "pair_stats.csv"), encoding="utf-8-sig")
    keep = ["pair_id", "dice_copy_aligned", "dice_copy_raw", "cov_soft",
            "cov_bin", "touch", "tgt_px", "disp_mm", "mmpx_curr", "native_w"]
    df = df.merge(st[keep], on="pair_id", how="left", validate="one_to_one")
    if df["mmpx_curr"].isna().any():
        raise RuntimeError("pair_stats.csv ile eslesmeyen cift var -> onbellegi yenileyin.")
    df["mm_per_px_512"] = df["mmpx_curr"] * df["native_w"] / float(IMG_SIZE)
    if os.name != "nt":
        for c in ("curr_img", "curr_mask", "prev_img", "prev_mask"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace("\\", "/", regex=False)
    return df, cdir, meta


def build_loaders(fold, csv_path=None, base_dir=None, cache_root=None,
                  batch_size=None, num_workers=None,
                  img_size=None, image_channels=None,
                  augment=True, use_prev_img=False, seed=None, preload=None):
    csv_path       = CSV_PATH       if csv_path is None else csv_path
    base_dir       = BASE_DIR       if base_dir is None else base_dir
    cache_root     = CACHE_ROOT     if cache_root is None else cache_root
    batch_size     = BATCH_SIZE     if batch_size is None else batch_size
    num_workers    = NUM_WORKERS    if num_workers is None else num_workers
    img_size       = IMG_SIZE       if img_size is None else img_size
    image_channels = IMAGE_CHANNELS if image_channels is None else image_channels
    seed           = SEED           if seed is None else seed
    preload        = PRELOAD        if preload is None else preload
    set_seed(seed)
    df, cdir, _ = load_table(csv_path, cache_root)

    train_df = df[df["fold"] != fold]
    val_df   = df[df["fold"] == fold]

    # --- sizinti: true_patient_id ASIL anahtar ---
    for key in ("true_patient_id", "patient_key"):
        if key in df.columns:
            ov = set(train_df[key]) & set(val_df[key])
            assert not ov, f"SIZINTI ({key})! Ortak: {sorted(ov)[:5]}"

    mk = lambda d, aug: TemporalPairDataset(
        d, base_dir, cdir, img_size, image_channels,
        augment=aug, use_prev_img=use_prev_img, preload=preload, seed=seed)

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(mk(train_df, augment), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False, drop_last=False,
                              generator=g, worker_init_fn=_worker_init)
    val_loader   = DataLoader(mk(val_df, False), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)
    return train_loader, val_loader


# ------------------------ Normalizasyon sabitleri --------------------
# Model sarmalayicisinda YALNIZCA RGB kanallarina uygulanir; prior kanali HAM.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
SAM2_MEAN     = (0.485, 0.456, 0.406)
SAM2_STD      = (0.229, 0.224, 0.225)


# ------------------------------ Saglik testi -------------------------
def sanity(fold=0):
    print(f"[i] Fold {fold}")
    tr, va = build_loaders(fold, augment=True)
    print(f"[i] train={len(tr.dataset)}  val={len(va.dataset)}")

    b = next(iter(tr))
    for k in ("image", "prior", "mask"):
        t = b[k]
        print(f"    {k:6s} {tuple(t.shape)} [{t.min():.3f},{t.max():.3f}]")
    print(f"    pozitif% = {100*b['mask'].mean():.4f}")

    # 1) augmentasyonsuz prior onbellekle birebir mi
    tr0, _ = build_loaders(fold, augment=False)
    ds = tr0.dataset
    s = ds[0]
    ref = ds._load_prior(ds.df.iloc[0]["pair_id"])
    assert np.allclose(s["prior"][0].numpy(), ref, atol=1e-6), "prior onbellekle uyusmuyor"
    print("[OK] augment=False -> prior onbellekle birebir")

    # 2) ortak geometrik donusum image/mask iliskisini koruyor mu
    rng = np.random.default_rng(0)
    im = np.zeros((64, 64), np.float32); im[20:40, 20:40] = 1.0
    pr = im.copy(); mk_ = im.copy()
    a2 = dict(AUG); a2["hflip_p"] = 1.0; a2["affine_p"] = 1.0
    i2, p2, m2 = _joint_affine(im, pr, mk_, rng, a2)
    inter = np.logical_and(i2 > 0.5, m2 > 0.5).sum()
    union = np.logical_or(i2 > 0.5, m2 > 0.5).sum()
    print(f"[OK] ortak affine sonrasi image/mask IoU = {inter/union:.4f} (>0.95 beklenir)")

    # 3) prior dropout orani
    ds2 = build_loaders(fold, augment=True)[0].dataset
    z = sum(1 for i in range(len(ds2)) if ds2[i]["prior"].max() == 0)
    print(f"[OK] prior dropout gozlenen oran = {z/len(ds2):.3f} "
          f"(hedef {AUG['prior_drop_p']})")

    # 4) taban cizgisi val setinde
    d = np.array([float(x) for x in va.dataset.df["dice_copy_aligned"]])
    print(f"[i] fold {fold} val taban cizgisi Dice = {d.mean():.4f}")


if __name__ == "__main__":
    sanity(0)
