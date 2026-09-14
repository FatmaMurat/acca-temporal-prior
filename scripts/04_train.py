# -*- coding: utf-8 -*-
"""
04_train.py
-----------
AC-CA calismasi - Adim 4: 5-fold egitim koşucusu (Colab icin).

KESINTIYE DAYANIKLILIK
  Colab oturumu her an kopabilir. Strateji katmanli, cunku tam kontrol
  noktasi ~1.1 GB (91M parametre + AdamW durumlari) ve bunu her epoch
  Drive'a yazmak egitimden uzun surer:

    her epoch      -> /content/ckpt/fold{k}_last.pt   (yerel disk, hizli)
    her epoch      -> Drive/metrics.csv               (birkac KB)
    iyilesince     -> Drive/fold{k}_best.pt           (yalniz model, fp16, ~182 MB)
    her SYNC_EVERY -> Drive/fold{k}_last.pt           (sigorta)

  Oturum koparsa run_all() kaldigi foldun kaldigi epoch'undan devam eder.
  Biten foldlar progress.json'dan okunup atlanir.

  DUZELTME: `best` degeri artik GUNCELLENDIKTEN SONRA kontrol noktasina
  yaziliyor. Onceki sirada devam eden oturum bir epoch bayat `best` ile
  basliyordu ve daha kotu bir epoch gercek en iyiyi ezebiliyordu.

KUCUK VERI ONLEMLERI (fold basina 194 cift)
  - Omurga ve kafa AYRI ogrenme orani. 87M onceden egitilmis omurgayi
    kafa ile ayni oranda guncellemek on-egitimi bozar.
  - Ilk FREEZE_EPOCHS boyunca omurga donuk: piramit ve decoder rastgele
    baslarken omurgaya gradyan gitmesin.
  - Isinma + kosinus schedule.

  DIKKAT (secenek 3): SAM2'nin prompt encoder'i da ONCEDEN EGITILMIS bir
  bilesendir. _BB_PREFIXES listesinde yer alir; aksi halde LR_HEAD (5e-4)
  ve WD 0.05 ile epoch 0'dan itibaren guncellenir ve secenek 3 yontem
  yuzunden degil, optimizer ayari yuzunden kaybeder.

ESIK: 0.5 SABIT. Val uzerinde esik aramak 5-fold CV'de val=test oldugu
icin iyimser yanlilik yaratir. Esik taramasi yalnizca tani amacli loglanir.
NOT: ayni gerekce EPOCH SECIMI icin de gecerlidir (en iyi epoch val'e gore
seciliyor). Duyarlilik analizi olarak sabit butceli protokol de raporlanir;
gerekli veri metrics.csv'de zaten var, ek GPU maliyeti yok.
"""

import os
import gc
import json
import time
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:                      # notebook mu, betik mi
    get_ipython()         # noqa
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False

# ================================ CONFIG ==============================
BASE_DIR    = "/content/acca"
CACHE_ROOT  = "/content/acca/_prior_cache"
SCRIPTS_DIR = "/content/acca/scripts"
LOCAL_CKPT  = "/content/ckpt"
DRIVE_DIR   = "/content/drive/MyDrive/acca_runs"

BACKBONE    = "sam2"           # "eva02" | "sam2"
SAM2_CKPT   = "/content/sam2.1_hiera_base_plus.pt"
RUN_NAME    = None             # None -> otomatik etiket (asagida)

PRIOR_MODE  = "channel"        # "channel" (4. kanal) | "prompt" (secenek 3)
PROMPT_GATE = True             # prompt modunda sifir baslatmali kapi

EPOCHS      = 100
BATCH_SIZE  = 4
ACCUM       = 1                # etkin batch = BATCH_SIZE * ACCUM
LR_HEAD     = 5e-4
LR_BACKBONE = 3e-5
WD          = 0.05
WARMUP_EP   = 5
FREEZE_EPOCHS = 3
GRAD_CLIP   = 1.0
AMP         = True
SYNC_EVERY  = 5
SEED        = 42

USE_PRIOR   = True             # False -> prior ablasyonu
USE_STEM    = True             # False -> HiResStem devre disi (cozunurluk ablasyonu)

# Onceden egitilmis / omurga sayilan parametre onekleri.
# backbone.bb        : EVA-02 (timm)
# backbone.enc       : SAM2 image encoder
# backbone.prompt_enc: SAM2 prompt encoder (SECENEK 3 - on-egitimli!)
# backbone.gate ve backbone.proj KASTEN disarida: bunlar yeni/rastgele.
_BB_PREFIXES = ("backbone.bb", "backbone.enc", "backbone.prompt_enc")
# ======================================================================


# ------------------------------ metrikler -----------------------------
def dice_iou_np(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    sp, sg = p.sum(), g.sum()
    if sp == 0 and sg == 0:
        return 1.0, 1.0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return (2.0 * inter / (sp + sg) if (sp + sg) else 0.0,
            inter / union if union else 0.0)


def hd95(pred, gt, spacing=1.0):
    """Bos tahmin veya bos hedefte NaN doner (ayrica SAYILIR - bkz. rec)."""
    from scipy.ndimage import distance_transform_edt as edt
    p, g = pred > 0.5, gt > 0.5
    if not p.any() or not g.any():
        return np.nan
    dg = edt(~g) * spacing
    dp = edt(~p) * spacing
    from scipy.ndimage import binary_erosion
    bp = p & ~binary_erosion(p)
    bg = g & ~binary_erosion(g)
    if not bp.any() or not bg.any():
        return np.nan
    d = np.concatenate([dg[bp], dp[bg]])
    return float(np.percentile(d, 95))


# ------------------------- kontrol noktasi I/O ------------------------
def atomic_save(obj, path):
    """Once .tmp'ye yaz, sonra yeniden adlandir. Yarim dosya kalmaz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_best_fp16(model, path):
    sd = {k: (v.half() if v.is_floating_point() else v)
          for k, v in model.state_dict().items()}
    atomic_save(sd, path)


def load_progress(run_dir):
    p = os.path.join(run_dir, "progress.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"done_folds": [], "run": os.path.basename(run_dir)}


def save_progress(run_dir, prog):
    with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)


# ----------------------------- optimizer ------------------------------
def make_optimizer(model):
    """Omurga (on-egitimli) ve kafa (yeni) AYRI ogrenme orani."""
    bb, head, bb_n, head_n = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(_BB_PREFIXES):
            bb.append(p); bb_n.append(n)
        else:
            head.append(p); head_n.append(n)
    print(f"    omurga param grubu: {len(bb)} (lr={LR_BACKBONE})  "
          f"kafa/piramit: {len(head)} (lr={LR_HEAD})")
    # secenek 3 sagligi: prompt encoder yanlis gruba dustuyse burada gorunur
    n_pe = sum(1 for n in bb_n if n.startswith("backbone.prompt_enc"))
    if any(n.startswith("backbone.prompt_enc") for n in head_n):
        raise RuntimeError("prompt_enc kafa grubuna dustu - _BB_PREFIXES bozuk.")
    if n_pe:
        print(f"    [secenek 3] prompt_enc {n_pe} tensor omurga grubunda")
    return torch.optim.AdamW(
        [{"params": bb,   "lr": LR_BACKBONE},
         {"params": head, "lr": LR_HEAD}], weight_decay=WD)


def lr_at(epoch, base):
    if epoch < WARMUP_EP:
        return base * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return base * 0.5 * (1 + np.cos(np.pi * t))


def set_backbone_frozen(model, frozen):
    """
    getattr(bb) or getattr(enc) kalibi kirilgandi (prompt_enc'i kacirir).
    Artik parametre adina gore, make_optimizer ile AYNI listeden.
    """
    n = 0
    for name, p in model.named_parameters():
        if name.startswith(_BB_PREFIXES):
            p.requires_grad = not frozen
            n += 1
    return n


# ------------------------------- egitim -------------------------------
def _prep_prior(pri):
    """USE_PRIOR=False ablasyonu, moda gore dogru karsiligi verir."""
    if USE_PRIOR:
        return pri
    if PRIOR_MODE == "prompt":
        return None            # SAM2'nin kendi no_mask_embed yolu
    return torch.zeros_like(pri)


def train_fold(fold, dl_mod, model_mod, run_dir, device):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)

    tr, va = dl_mod.build_loaders(fold, batch_size=BATCH_SIZE, augment=True)
    print(f"  fold {fold}: train={len(tr.dataset)} val={len(va.dataset)}")

    model = model_mod.build_model(
        BACKBONE, pretrained=True, use_stem=USE_STEM,
        prior_mode=PRIOR_MODE, prompt_gate=PROMPT_GATE,
        sam2_ckpt=SAM2_CKPT if BACKBONE == "sam2" else None).to(device)
    if not USE_STEM:
        print("  [ablasyon] HiResStem DEVRE DISI")
    if PRIOR_MODE == "prompt":
        print("  [secenek 3] prior MASKE PROMPT yolundan, girdi 3 kanal")
    crit = model_mod.ComboLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    local_last = os.path.join(LOCAL_CKPT, f"fold{fold}_last.pt")
    drive_last = os.path.join(run_dir, f"fold{fold}_last.pt")
    drive_best = os.path.join(run_dir, f"fold{fold}_best.pt")

    start_ep, best = 0, -1.0
    ck = None
    resume_from = local_last if os.path.exists(local_last) else (
        drive_last if os.path.exists(drive_last) else None)
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"  [devam] {os.path.basename(resume_from)} -> epoch {start_ep}, "
              f"en iyi Dice {best:.4f}")

    n_fr = set_backbone_frozen(model, start_ep < FREEZE_EPOCHS)
    print(f"    omurga tensor sayisi: {n_fr}")
    opt = make_optimizer(model)
    if ck is not None and "opt" in ck:
        try:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
        except Exception as e:
            print(f"  [not] optimizer durumu yuklenemedi ({type(e).__name__}), sifirdan")
    del ck

    base_dice = float(np.mean(va.dataset.df["dice_copy_aligned"]))
    print(f"  fold {fold} taban cizgisi (prior kopyala) Dice = {base_dice:.4f}")

    mcsv = os.path.join(run_dir, "metrics.csv")
    for ep in range(start_ep, EPOCHS):
        if ep == FREEZE_EPOCHS and start_ep <= FREEZE_EPOCHS:
            set_backbone_frozen(model, False)
            opt = make_optimizer(model)
            print(f"  [epoch {ep}] omurga cozuldu")

        for g, base in zip(opt.param_groups, (LR_BACKBONE, LR_HEAD)):
            g["lr"] = lr_at(ep, base)

        # ---- egitim ----
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr):
            img = b["image"].to(device, non_blocking=True)
            pri = _prep_prior(b["prior"].to(device, non_blocking=True))
            msk = b["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                out = model(img, pri)
                loss, log = crit(out.float(), msk, epoch=ep, total=EPOCHS)
            scaler.scale(loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1

        # ---- degerlendirme ----
        model.eval()
        D, I, H, rows = [], [], [], []
        with torch.no_grad():
            for b in va:
                img = b["image"].to(device)
                pri = _prep_prior(b["prior"].to(device))
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    p = torch.sigmoid(model(img, pri).float()).cpu().numpy()
                g = b["mask"].numpy()
                for j in range(p.shape[0]):
                    d, iou = dice_iou_np(p[j, 0], g[j, 0])
                    h = hd95(p[j, 0] > 0.5, g[j, 0] > 0.5)
                    D.append(d); I.append(iou); H.append(h)
                    rows.append(dict(pair_id=b["pair_id"][j], dice=d, iou=iou,
                                     hd95=h, dice_copy=float(b["dice_copy"][j])))

        mD, mI = float(np.mean(D)), float(np.mean(I))
        H = np.asarray(H, dtype=float)
        n_nan = int(np.isnan(H).sum())
        mH = float(np.nanmean(H)) if n_nan < len(H) else float("nan")
        # HD95 tanimsiz cift sayisi RAPORLANIR: yoksa her kol farkli bir alt
        # kume uzerinde HD95 bildirir ve karsilastirma eslesmis olmaz.
        rec = dict(fold=fold, epoch=ep, loss=tot / max(1, nb),
                   val_dice=mD, val_iou=mI, val_hd95=mH,
                   hd95_undefined=n_nan, n_val=len(H),
                   baseline_dice=base_dice, delta=mD - base_dice,
                   lr_bb=opt.param_groups[0]["lr"], sec=time.time() - t0,
                   ft=log["ft"], bd=log["bd"], w=log["w"])
        if PRIOR_MODE == "prompt" and PROMPT_GATE:
            rec["gate"] = float(model.backbone.gate.detach().float().item())
        pd.DataFrame([rec]).to_csv(mcsv, mode="a", index=False,
                                   header=not os.path.exists(mcsv))
        gtxt = f" gate {rec['gate']:+.3f}" if "gate" in rec else ""
        print(f"    ep {ep:3d} loss {rec['loss']:.4f} | Dice {mD:.4f} "
              f"(taban {base_dice:.4f}, fark {mD-base_dice:+.4f}) "
              f"IoU {mI:.4f} HD95 {mH:.1f} (tanimsiz {n_nan}){gtxt} "
              f"| {rec['sec']:.0f}s")

        # ---- kayit: once best guncelle, SONRA state yaz ----
        if mD > best:
            best = mD
            save_best_fp16(model, drive_best)
            pd.DataFrame(rows).to_csv(
                os.path.join(run_dir, f"fold{fold}_val_pairs.csv"),
                index=False, encoding="utf-8-sig")
        state = dict(model=model.state_dict(), opt=opt.state_dict(),
                     scaler=scaler.state_dict(), epoch=ep, best=best, fold=fold)
        atomic_save(state, local_last)
        if (ep + 1) % SYNC_EVERY == 0:
            shutil.copy2(local_last, drive_last)
            print(f"    [senkron] {drive_last}")

    del model, opt, scaler
    gc.collect(); torch.cuda.empty_cache()
    return best, base_dice


# ------------------------------- koşucu -------------------------------
def run_tag():
    t = f"{BACKBONE}_{'prior' if USE_PRIOR else 'noprior'}"
    if PRIOR_MODE == "prompt":
        t += "_prompt" + ("" if PROMPT_GATE else "_nogate")
    if not USE_STEM:
        t += "_nostem"
    return t


def run_all(folds=(0, 1, 2, 3, 4)):
    import importlib.util, sys

    def load(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s); sys.modules[name] = m
        s.loader.exec_module(m); return m

    dl_mod = load("dl", os.path.join(SCRIPTS_DIR, "temporal_pair_dataloader_v2.py"))
    # DIKKAT: modul degiskenine atama yetmez, configure() cagrilmali
    dl_mod.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT,
                     batch_size=BATCH_SIZE, seed=SEED)
    model_mod = load("mm", os.path.join(SCRIPTS_DIR, "03_model.py"))

    run = RUN_NAME or run_tag()
    run_dir = os.path.join(DRIVE_DIR, run)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(LOCAL_CKPT, exist_ok=True)
    device = torch.device("cuda")
    print(f"[i] kosu: {run}\n[i] Drive: {run_dir}\n[i] GPU: "
          f"{torch.cuda.get_device_name(0)}")
    print(f"[i] omurga={BACKBONE}  prior={USE_PRIOR}  mod={PRIOR_MODE}  "
          f"stem={USE_STEM}  epoch={EPOCHS}")

    prog = load_progress(run_dir)
    res = []
    for k in folds:
        if k in prog["done_folds"]:
            print(f"[atla] fold {k} zaten bitti")
            continue
        print(f"\n{'='*60}\nFOLD {k}\n{'='*60}")
        best, base = train_fold(k, dl_mod, model_mod, run_dir, device)
        res.append((k, best, base))
        prog["done_folds"].append(k)
        save_progress(run_dir, prog)

    print(f"\n{'='*60}\nOZET: {run}\n{'='*60}")
    if res:
        b = np.array([r[1] for r in res]); c = np.array([r[2] for r in res])
        for k, bb, cc in res:
            print(f"  fold {k}: Dice {bb:.4f}  taban {cc:.4f}  fark {bb-cc:+.4f}")
        print(f"  ORTALAMA Dice {b.mean():.4f} +/- {b.std():.4f}")
        print(f"  TABAN    Dice {c.mean():.4f}")
        print(f"  KAZANC   {b.mean()-c.mean():+.4f}")
    else:
        print("  tum foldlar zaten bitmis. metrics.csv'ye bakin.")


def smoke_test(n=10, epochs=30):
    """10 cift uzerinde kasten asiri ogrenme. Dice ~0.9'a cikmali."""
    global EPOCHS, FREEZE_EPOCHS, WARMUP_EP, SYNC_EVERY, RUN_NAME
    EPOCHS, FREEZE_EPOCHS, WARMUP_EP, SYNC_EVERY = epochs, 0, 1, 10**6
    RUN_NAME = "SMOKE_" + run_tag()
    import importlib.util, sys
    s = importlib.util.spec_from_file_location(
        "dl", os.path.join(SCRIPTS_DIR, "temporal_pair_dataloader_v2.py"))
    dl = importlib.util.module_from_spec(s); sys.modules["dl"] = dl
    s.loader.exec_module(dl)
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT, batch_size=BATCH_SIZE)
    orig = dl.build_loaders

    def small(fold, **kw):
        kw["augment"] = False
        tr, va = orig(fold, **kw)
        tr.dataset.df = tr.dataset.df.head(n).reset_index(drop=True)
        va.dataset.df = tr.dataset.df.copy()
        return tr, va
    dl.build_loaders = small
    print(f"[duman testi] {n} cift, {epochs} epoch, augmentasyon kapali")
    print("[duman testi] Dice 0.85 ustune cikmazsa boru hattinda hata var")
    run_all(folds=(0,))


# NOT: exec() ile yuklendiginde notebook'ta __name__ == "__main__" oldugu icin
# otomatik calistirma KALDIRILDI. Yukledikten sonra elle cagirin:
#
#   SECENEK 3:
#     T.BACKBONE   = "sam2"
#     T.PRIOR_MODE = "prompt"
#     T.USE_PRIOR  = True
#     T.PROMPT_GATE = True
#     T.SYNC_EVERY = 5
#     T.smoke_test(n=10, epochs=30)   # once bu -> sam2_prior_prompt smoke
#     T.run_all()                     # sonra bu -> sam2_prior_prompt
if __name__ == "__main__" and not _IN_NOTEBOOK:
    run_all()
