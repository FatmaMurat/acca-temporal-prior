# -*- coding: utf-8 -*-
"""
15_last_epoch.py
----------------
AC-CA - Adim 15: esit kontrol noktasi protokolu (on-degerlendirme madde 1.6).

SORUN
  Bes katli capraz dogrulamada dogrulama kumesi ayni zamanda test kumesidir.
  Bizim kollarimiz EN IYI epoch'un agirliklariyla raporlaniyor (fold*_best.pt,
  fold*_val_pairs.csv); nnU-Net ise 11_nnunet_baseline.py'de `nnUNetv2_predict`
  ile, yani nnU-Net'in VARSAYILAN checkpoint_final.pth (SON epoch) dosyasiyla
  tahmin ediyor. Karsilastirma bu haliyle bizim lehimize egik.

  metrics.csv yalnizca fold duzeyinde ortalama Dice tutar (Tablo S2). Asil
  soru ortalama degil KUYRUK: makalenin nnU-Net anlatisi "ortalama esit ama
  tam basarisizlik 20'ye 6, bos tahmin 14'e 1" uzerine kurulu. Bu sayilar
  cift duzeyinde tahmin gerektirir, dolayisiyla SON epoch agirliklariyla
  yeniden cikarim yapilmalidir.

BU SCRIPT NE YAPAR
  1) run_inference(): her kosunun fold{k}_last.pt dosyasini yukler (SYNC_EVERY=5
     ve EPOCHS=100 oldugu icin bu dosya epoch 99 durumudur; kontrol edilir),
     o foldun dogrulama ciftlerinde cikarim yapar ve cift bazli metrikleri
     fold{k}_val_pairs_last.csv olarak yazar. Egitim YOK.
  2) report(): en iyi epoch ve son epoch protokollerini yan yana koyar;
     Dice, tam basarisizlik (Dice = 0) ve bos tahmin sayilarini verir;
     her kolu nnU-Net ile SON epoch protokolunde eslesmis olarak karsilastirir
     (Wilcoxon + hasta duzeyinde bootstrap; basarisizlik sayilari icin kesin
     McNemar).

KARAR
  Kuyruk farki (bos tahmin ve Dice = 0) son epoch'ta da duruyorsa makalenin
  anlatisi ayakta kalir ve birincil protokol SON EPOCH yapilabilir; en iyi
  epoch duyarlilik analizine dusurulur. Durmuyorsa nnU-Net karsilastirmasinin
  ifadesi degistirilmelidir.

Kullanim (Colab):
    L = S.load("L15", "/content/acca/scripts/15_last_epoch.py")
    L.run_inference()          # GPU, ~10-15 dk (kontrol noktalari buyuk)
    L.report()                 # CPU
"""

import os
import sys
import glob
import time
import importlib.util

import numpy as np
import pandas as pd

# ================================ CONFIG ==============================
BASE_DIR   = "/content/acca"
CACHE_ROOT = "/content/acca/_prior_cache"
SCRIPTS    = "/content/acca/scripts"
RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
OUT_DIR    = "/content/drive/MyDrive/acca_runs/_protokol"

CKPT       = "/content/sam2.1_hiera_base_plus.pt"
CFG        = "configs/sam2.1/sam2.1_hiera_b+.yaml"
IMG_SIZE   = 512
EPOCHS     = 100                   # 04_train.EPOCHS ile ayni
N_BOOT     = 2000
SEED       = 42

# kosu adi -> build_model argumanlari + prior'in nasil verilecegi
# (04_train._prep_prior ile AYNI: USE_PRIOR=False ise channel modda sifir,
#  prompt modda None)
ARMS = {
    "sam2_prior":           dict(which="sam2", use_prior=True,  kw=dict()),
    "sam2_prior_lowres32":  dict(which="sam2", use_prior=True,  kw=dict(prior_grid=32)),
    "sam2_prior_binary":    dict(which="sam2", use_prior=True,  kw=dict(prior_binary=True)),
    "sam2_prior_prompt":    dict(which="sam2", use_prior=True,  kw=dict(prior_mode="prompt")),
    "sam2_noprior":         dict(which="sam2", use_prior=False, kw=dict()),
    "eva02_prior":          dict(which="eva02", use_prior=True,  kw=dict()),
    "eva02_noprior":        dict(which="eva02", use_prior=False, kw=dict()),
}
# nnU-Net zaten checkpoint_final (son epoch) ile tahmin ediyor; yeniden
# cikarim gerekmez, mevcut fold*_val_pairs.csv dosyalari kullanilir.
NNUNET = "nnunet_prior"
# ======================================================================


def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print(t); print("=" * 72)


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def _helpers():
    L = _load("L12b", os.path.join(SCRIPTS, "12_lowres_prior.py"))
    L.RUNS_DIR = RUNS_DIR
    return L


def dice_iou(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    sp, sg = p.sum(), g.sum()
    if sp == 0 and sg == 0:
        return 1.0, 1.0
    inter = np.logical_and(p, g).sum(); union = np.logical_or(p, g).sum()
    return (2.0 * inter / (sp + sg) if (sp + sg) else 0.0, inter / union if union else 0.0)


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt as edt, binary_erosion
    p, g = pred > 0.5, gt > 0.5
    if not p.any() or not g.any():
        return np.nan
    bp = p & ~binary_erosion(p); bg = g & ~binary_erosion(g)
    if not bp.any() or not bg.any():
        return np.nan
    return float(np.percentile(np.concatenate([edt(~g)[bp], edt(~p)[bg]]), 95))


# ======================================================================
# 1) SON EPOCH AGIRLIKLARIYLA CIKARIM
# ======================================================================
def run_inference(runs=None, folds=(0, 1, 2, 3, 4), ckpt=CKPT, batch_size=2,
                  overwrite=False):
    import torch
    dl = _load("dl15", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    MM = _load("MM15", os.path.join(SCRIPTS, "03_model.py"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = list(runs or ARMS)

    hr("SON EPOCH AGIRLIKLARIYLA CIKARIM")
    print(f"  cihaz: {dev}   kosu sayisi: {len(runs)}")
    t0 = time.time()
    for run in runs:
        if run not in ARMS:
            print(f"  [!] {run}: ARMS sozlugunde yok, atlaniyor"); continue
        cfgA = ARMS[run]
        rd = os.path.join(RUNS_DIR, run)
        print(f"\n  {run}")
        for k in folds:
            outp = os.path.join(rd, f"fold{k}_val_pairs_last.csv")
            if os.path.exists(outp) and not overwrite:
                print(f"    fold {k}: zaten var, atlaniyor"); continue
            w = os.path.join(rd, f"fold{k}_last.pt")
            if not os.path.exists(w):
                print(f"    [!] fold {k}: fold{k}_last.pt YOK"); continue
            ck = torch.load(w, map_location="cpu", weights_only=False)
            ep = int(ck.get("epoch", -1))
            if ep != EPOCHS - 1:
                print(f"    [!] fold {k}: kontrol noktasi epoch {ep}, beklenen {EPOCHS-1} "
                      f"- bu fold icin SON EPOCH degil, atlaniyor")
                del ck; continue
            model = MM.build_model(cfgA["which"], img_size=IMG_SIZE, pretrained=False,
                                   sam2_ckpt=ckpt if cfgA["which"] == "sam2" else None,
                                   sam2_cfg=CFG, **cfgA["kw"]).to(dev)
            model.load_state_dict(ck["model"])
            del ck
            model.eval()
            _, va = dl.build_loaders(k, batch_size=batch_size, augment=False)
            rows = []
            with torch.no_grad():
                for b in va:
                    img = b["image"].to(dev)
                    pri = b["prior"].to(dev)
                    if not cfgA["use_prior"]:
                        pri = None if cfgA["kw"].get("prior_mode") == "prompt" \
                            else torch.zeros_like(pri)
                    if dev.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            q = torch.sigmoid(model(img, pri).float()).cpu().numpy()
                    else:
                        q = torch.sigmoid(model(img, pri).float()).cpu().numpy()
                    gt = b["mask"].numpy()
                    for j in range(q.shape[0]):
                        dsc, iou = dice_iou(q[j, 0], gt[j, 0])
                        rows.append(dict(pair_id=b["pair_id"][j], dice=dsc, iou=iou,
                                         hd95=hd95(q[j, 0] > 0.5, gt[j, 0] > 0.5),
                                         empty_pred=int((q[j, 0] > 0.5).sum() == 0),
                                         dice_copy=float(b["dice_copy"][j]), epoch=ep))
            pd.DataFrame(rows).to_csv(outp, index=False, encoding="utf-8-sig")
            print(f"    fold {k}: epoch {ep}, {len(rows)} cift, "
                  f"Dice {np.mean([r['dice'] for r in rows]):.4f}  ({time.time()-t0:.0f} sn)")
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\n  bitti ({time.time()-t0:.0f} sn)")


# ======================================================================
# 2) PROTOKOL KARSILASTIRMASI
# ======================================================================
def _pairs(run, suffix=""):
    fs = sorted(glob.glob(os.path.join(RUNS_DIR, run, f"fold*_val_pairs{suffix}.csv")))
    if not fs:
        return None
    out = []
    for f in fs:
        x = pd.read_csv(f, encoding="utf-8-sig")
        x["fold"] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        out.append(x)
    P = pd.concat(out, ignore_index=True)
    if "empty_pred" not in P.columns:          # en iyi epoch dosyalari: HD95 = NaN vekil
        P["empty_pred"] = P.hd95.isna().astype(int)
    return P


def _mcnemar(a_fail, b_fail):
    """Kesin McNemar (iki yonlu). a, b: ikili diziler (True = basarisiz)."""
    from scipy.stats import binomtest
    n01 = int((a_fail & ~b_fail).sum()); n10 = int((~a_fail & b_fail).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


def report(runs=None, n_boot=N_BOOT, save=True):
    L = _helpers()
    runs = list(runs or ARMS)
    hr("PROTOKOL KARSILASTIRMASI (en iyi epoch / son epoch)")

    # ---------------- 1) kol bazinda iki protokol ----------------
    rows, last = [], {}
    for run in runs + [NNUNET]:
        B = _pairs(run, "")
        Lp = _pairs(run, "_last") if run != NNUNET else B   # nnU-Net zaten son epoch
        for prot, P in (("en iyi epoch", B), ("son epoch", Lp)):
            if P is None:
                continue
            if run == NNUNET and prot == "en iyi epoch":
                continue                                    # tek protokolu var
            fm = P.groupby("fold").dice.mean()
            rows.append(dict(kol=run, protokol=prot, n=len(P),
                             Dice=P.dice.mean(),
                             fold_ort=fm.mean(), fold_sd=fm.std(ddof=1),
                             sifir=int((P.dice == 0).sum()),
                             bos=int(P.empty_pred.sum())))
        if Lp is not None:
            last[run] = Lp[["pair_id", "fold", "dice", "hd95", "empty_pred"]]
    T1 = pd.DataFrame(rows)
    disp = T1.copy()
    disp["Dice"] = T1.Dice.map(lambda v: f"{v:.4f}")
    disp["fold ort. +/- SD"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(T1.fold_ort, T1.fold_sd)]
    print(disp[["kol", "protokol", "n", "Dice", "fold ort. +/- SD", "sifir", "bos"]]
          .to_string(index=False))
    print("  sifir: tam basarisizlik (Dice = 0)   bos: tamamen bos tahmin")
    print("  NOT: nnU-Net nnUNetv2_predict varsayilani (checkpoint_final) ile "
          "tahmin ettigi icin zaten SON epoch protokolundedir.")

    # ---------------- 2) son epoch protokolunde nnU-Net karsilastirmasi -----
    hr("SON EPOCH PROTOKOLUNDE nnU-NET KARSILASTIRMASI")
    if NNUNET not in last:
        print(f"  [!] {NNUNET} cift dosyalari yok; karsilastirma atlandi")
        T2 = None
    else:
        N = last[NNUNET].rename(columns={"dice": "dice_nn", "empty_pred": "bos_nn"})
        dlm = _load("dl15b", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
        dlm.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
        meta, _, _ = dlm.load_table()
        pid = "true_patient_id" if "true_patient_id" in meta.columns else "patient_key"
        meta = meta[["pair_id", pid]].rename(columns={pid: "patient"})
        rng = np.random.default_rng(SEED)
        rows = []
        for run in runs:
            if run not in last:
                continue
            A = last[run].rename(columns={"dice": "dice_a", "empty_pred": "bos_a"})
            M = (meta.merge(A[["pair_id", "dice_a", "bos_a"]], on="pair_id")
                     .merge(N[["pair_id", "dice_nn", "bos_nn"]], on="pair_id"))
            up, inv = np.unique(M.patient.astype(str).values, return_inverse=True)
            C = np.bincount(inv).astype(float)
            IDX = rng.integers(0, len(up), size=(n_boot, len(up)))
            cnt = C[IDX].sum(1)
            bm = lambda v: np.bincount(inv, weights=np.asarray(v, float))[IDX].sum(1) / cnt
            lo, hi = np.percentile(bm(M.dice_a) - bm(M.dice_nn), [2.5, 97.5])
            f_a = (M.dice_a == 0).values; f_n = (M.dice_nn == 0).values
            n01, n10, p_f = _mcnemar(f_a, f_n)
            e_a = M.bos_a.astype(bool).values; e_n = M.bos_nn.astype(bool).values
            m01, m10, p_e = _mcnemar(e_a, e_n)
            rows.append(dict(kol=run, n=len(M),
                             dDice=M.dice_a.mean() - M.dice_nn.mean(), ci_lo=lo, ci_hi=hi,
                             p_dice=L._wilcoxon_p(M.dice_a, M.dice_nn),
                             sifir=f"{int(f_a.sum())} / {int(f_n.sum())}",
                             p_sifir=p_f,
                             bos=f"{int(e_a.sum())} / {int(e_n.sum())}", p_bos=p_e))
        T2 = pd.DataFrame(rows)
        disp = T2.copy()
        disp["dDice (95% GA)"] = [f"{o:+.4f} [{l:+.4f}, {h:+.4f}]"
                                  for o, l, h in zip(T2.dDice, T2.ci_lo, T2.ci_hi)]
        for c in ("p_dice", "p_sifir", "p_bos"):
            disp[c] = T2[c].map(L._fmt_p)
        print(disp[["kol", "n", "dDice (95% GA)", "p_dice", "sifir", "p_sifir",
                    "bos", "p_bos"]].to_string(index=False))
        print("  sifir / bos sutunlari: kol / nnU-Net. p degerleri kesin McNemar "
              "(eslesmis, ayrik ciftler uzerinden).")

    hr("YORUM")
    print("""  Makalenin nnU-Net anlatisi ortalamaya degil kuyruga dayaniyor:
  "ortalama esit, ama tam basarisizlik 20'ye 6 ve bos tahmin 14'e 1".
  Yukaridaki SON EPOCH tablosunda bu iki fark hala duruyorsa:
     - birincil protokol SON EPOCH yapilabilir (en iyi epoch duyarlilik analizi),
     - nnU-Net karsilastirmasi artik esit protokolde kurulur,
     - Tablo 2 ve ozet sayilari son epoch degerleriyle guncellenir.
  Fark kuculuyor ya da kayboluyorsa iddia zayiflatilmali ve bu acikca yazilmali.""")

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        T1.to_csv(os.path.join(OUT_DIR, "protokol_genel.csv"), index=False,
                  encoding="utf-8-sig")
        if T2 is not None:
            T2.to_csv(os.path.join(OUT_DIR, "protokol_nnunet.csv"), index=False,
                      encoding="utf-8-sig")
        print(f"\n  [i] kaydedildi: {OUT_DIR}")
    return T1, T2


# ======================================================================
# 3) IKI KOL ARASINDA ESLESMIS KARSILASTIRMA (protokol secilebilir)
# ======================================================================
def compare(run_a, run_b, protocol="son", n_boot=N_BOOT):
    """
    Iki kolu SECILEN protokolde karsilastirir: ortalama Dice farki (hasta
    duzeyinde bootstrap GA, Wilcoxon), tam basarisizlik ve bos tahmin (kesin
    McNemar) ve karar alt gruplari. protocol: "son" | "en iyi".
    nnU-Net her iki durumda da kendi (son epoch) dosyalarini kullanir.
    """
    L = _helpers()
    suf = "_last" if protocol == "son" else ""
    get = lambda r: _pairs(r, "" if r == NNUNET else suf)
    A, B = get(run_a), get(run_b)
    if A is None or B is None:
        print(f"  [-] cift dosyalari eksik ({run_a}: {A is not None}, {run_b}: {B is not None})")
        return None
    dlm = _load("dl15c", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dlm.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    meta, _, _ = dlm.load_table()
    pid = "true_patient_id" if "true_patient_id" in meta.columns else "patient_key"
    keep = ["pair_id", pid, "tgt_px"] + [c for c in ("touch", "cov_soft") if c in meta.columns]
    M = (meta[keep].rename(columns={pid: "patient"})
         .merge(A[["pair_id", "dice", "empty_pred"]].rename(columns={"dice": "a", "empty_pred": "ea"}), on="pair_id")
         .merge(B[["pair_id", "dice", "empty_pred"]].rename(columns={"dice": "b", "empty_pred": "eb"}), on="pair_id"))

    hr(f"{run_a}  vs  {run_b}   ({protocol} epoch, n = {len(M)})")
    rng = np.random.default_rng(SEED)
    up, inv = np.unique(M.patient.astype(str).values, return_inverse=True)
    IDX = rng.integers(0, len(up), size=(n_boot, len(up)))

    def boot(mask):
        m = np.asarray(mask, float)
        d = (M.a - M.b).values * m
        S = np.bincount(inv, weights=d); N = np.bincount(inv, weights=m)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = S[IDX].sum(1) / N[IDX].sum(1)
        return np.nanpercentile(v, [2.5, 97.5])

    subs = [("tum ciftler", np.ones(len(M), bool))]
    if "touch" in M.columns:
        subs.append(("ortusme yok", ~M.touch.astype(bool).values))
    if "cov_soft" in M.columns:
        subs.append(("kapsama Q1", (M.cov_soft <= M.cov_soft.quantile(0.25)).values))
    subs += [("< 200 px", (M.tgt_px < 200).values), ("200-800 px", ((M.tgt_px >= 200) & (M.tgt_px < 800)).values),
             ("800-2000 px", ((M.tgt_px >= 800) & (M.tgt_px <= 2000)).values), ("> 2000 px", (M.tgt_px > 2000).values)]
    rows = []
    for lab, m in subs:
        if m.sum() < 5:
            continue
        lo, hi = boot(m)
        rows.append(dict(grup=lab, n=int(m.sum()), a=M.a[m].mean(), b=M.b[m].mean(),
                         fark=(M.a - M.b)[m].mean(), lo=lo, hi=hi,
                         p=L._wilcoxon_p(M.a[m], M.b[m])))
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    T.loc[T.index[1:], "p_holm"] = L._holm(T.p.values[1:])
    disp = T.copy()
    disp["a"] = T.a.map(lambda v: f"{v:.4f}"); disp["b"] = T.b.map(lambda v: f"{v:.4f}")
    disp["fark (95% GA)"] = [f"{o:+.4f} [{l:+.4f}, {h:+.4f}]" for o, l, h in zip(T.fark, T.lo, T.hi)]
    disp["p"] = T.p.map(L._fmt_p); disp["p_holm"] = T.p_holm.map(L._fmt_p)
    disp = disp.rename(columns={"a": run_a[:18], "b": run_b[:18]})
    print(disp[["grup", "n", run_a[:18], run_b[:18], "fark (95% GA)", "p", "p_holm"]].to_string(index=False))
    print("  p_holm: alt gruplar kendi aralarinda (tum ciftler satiri haric).")

    fa, fb = (M.a == 0).values, (M.b == 0).values
    n01, n10, pf = _mcnemar(fa, fb)
    ea, eb = M.ea.astype(bool).values, M.eb.astype(bool).values
    m01, m10, pe = _mcnemar(ea, eb)
    print(f"\n  tam basarisizlik: {int(fa.sum())} / {int(fb.sum())}   ayrik {n01} / {n10}   McNemar p = {L._fmt_p(pf)}")
    print(f"  bos tahmin      : {int(ea.sum())} / {int(eb.sum())}   ayrik {m01} / {m10}   McNemar p = {L._fmt_p(pe)}")
    return T


if __name__ == "__main__":
    report()
