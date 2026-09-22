# -*- coding: utf-8 -*-
"""
14_binary_prior.py
------------------
AC-CA - Adim 14: ikili prior ablasyonu (on-degerlendirme madde 1.5).

SORU
  Makalenin BIRINCI katkisi ve "ilk kez" iddiasi, prior'i ikili maske yerine
  surekli olasilik haritasi olarak kodlamaya dayaniyor. Bu simdiye kadar test
  edilmedi; metindeki "ikili prior yanlis konumu kesinlikle bildirirdi" cumlesi
  bir varsayim. Iki olcum yapilir:

  A) TAU TARAMASI (egitim YOK, saatler degil dakikalar)
     Model tau 10.7-21.4 mm augmentasyonuyla egitildigi icin mevcut
     fold*_best.pt agirliklariyla test zamaninda tau degistirilebilir:
         prior^g = exp(-d*g/tau)  ->  tau_eff = tau / g      (plato 1'de kalir)
     Ek olarak g -> sonsuz limiti, yani esikleme, IKILI prior'a karsilik gelir.
     Bu tarama modelin prior yumusakligina duyarliligini olcer ama ikili
     prior'la EGITILMIS bir modeli temsil etmez (dagitim disi girdi).

  B) IKILI KOL (egitim VAR, ~6 saat)
     Mimari, parametre sayisi, fold, seed ayni; prior aga girmeden esiklenir
     (03_model.py: binarize_prior). Asil karsilastirma budur.

YORUM CERCEVESI (sonuclar gorulmeden ONCE yazildi)
  a) ikili ~ yumusak  -> surekli kodlamanin katkisi yok. Birinci katki ve
     Sonuc'taki "ilk kez" cumlesi metinden CIKARILMALI; yontem yine de
     gecerlidir, yalnizca yenilik iddiasi dusem.
  b) ikili < yumusak, fark ortusmeyen 27 ciftte ve dusuk kapsama diliminde
     toplaniyor  -> makalenin mevcut anlatisi dogrulanir; Tablo 5'teki
     alt grup analizi ikili kolla tekrarlanir.
  c) ikili > yumusak  -> beklenmedik; kuyruk (Dice = 0) davranisina ve kucuk
     lezyonlara bakilir, iddia tersine cevrilir.
  Karar olcutu ortalama Dice degil, ortusme olmayan alt gruptur: surekli
  kodlamanin gerekcesi oradaki davranistir.

ADIMLAR (Colab)
  0) 13_colab_session.prepare(run="sam2_prior_binary")
  1) L.tau_sweep()    GPU, ~10 dk. Egitim yok. tau = 15 satiri makaledeki
                      0.7384 ortalamasini yeniden uretmeli (boru hatti kontrolu).
  2) L.verify()       GPU, ~3 dk. Esiklemenin forward icinde, omurga ve
                      govdeden ONCE yapildigini sinar.
  3) Egitim (04_train.py, PRIOR_BINARY=True) -> acca_runs/sam2_prior_binary
  4) L.report()       CPU. Eslesmis karsilastirma, alt gruplar, tabakalar,
                      hasta duzeyinde bootstrap, epoch protokolleri.

CIKTILAR: acca_runs/_binary/
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
OUT_DIR    = "/content/drive/MyDrive/acca_runs/_binary"

CKPT       = "/content/sam2.1_hiera_base_plus.pt"
CFG        = "configs/sam2.1/sam2.1_hiera_b+.yaml"
IMG_SIZE   = 512
TAU_MM     = 15.0                  # onbellekteki prior'in tau'su (01 ile ayni)

RUN_FULL   = "sam2_prior"
RUN_BIN    = "sam2_prior_binary"
RUN_PROMPT = "sam2_prior_prompt"
RUN_NOPRI  = "sam2_noprior"

# test zamani tau taramasi (mm). Egitim araligi 10.7-21.4; disina da bakilir.
TAUS       = [5.0, 7.5, 10.0, 15.0, 22.5, 30.0, 45.0]
N_BOOT     = 2000
SEED       = 42
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
    """12_lowres_prior.py'deki ortak analiz yardimcilarini yeniden kullanir."""
    L = _load("L12", os.path.join(SCRIPTS, "12_lowres_prior.py"))
    L.RUNS_DIR = RUNS_DIR
    return L


def _modules(need_model=True):
    dl = _load("dl14", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    MM = None
    if need_model:
        MM = _load("MM14", os.path.join(SCRIPTS, "03_model.py"))
        if not hasattr(MM, "binarize_prior"):
            raise RuntimeError("scripts/03_model.py ESKI surum (binarize_prior yok). "
                               "Guncel dosyayi kopyalayin.")
    return dl, MM


def _dice(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    sp, sg = p.sum(), g.sum()
    if sp == 0 and sg == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / (sp + sg)) if (sp + sg) else 0.0


# ======================================================================
# A) TEST ZAMANI TAU TARAMASI  (egitim yok)
# ======================================================================
def tau_sweep(run=RUN_FULL, taus=TAUS, ckpt=CKPT, folds=(0, 1, 2, 3, 4),
              batch_size=2, save=True):
    """
    Mevcut fold*_best.pt agirliklariyla prior'in yumusakligini test zamaninda
    degistirir. Her cift, o cifti dogrulamada goren foldun agirliklariyla
    degerlendirilir (sizinti yok).
    """
    import torch
    dl, MM = _modules()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gammas = [(f"tau={t:g}", TAU_MM / float(t)) for t in taus]

    hr(f"TEST ZAMANI TAU TARAMASI  ({run}, {len(taus)} tau + ikili)")
    print(f"  cihaz: {dev}   prior^g, g = {TAU_MM:g}/tau   (plato 1'de kalir)")
    rows, t0 = [], time.time()
    for k in folds:
        w = os.path.join(RUNS_DIR, run, f"fold{k}_best.pt")
        if not os.path.exists(w):
            print(f"  [!] fold {k}: agirlik yok, atlaniyor")
            continue
        _, va = dl.build_loaders(k, batch_size=batch_size, augment=False)
        model = MM.build_model("sam2", img_size=IMG_SIZE, pretrained=False,
                               sam2_ckpt=ckpt, sam2_cfg=CFG).to(dev)
        model.load_state_dict({q: v.float() for q, v in
                               torch.load(w, map_location=dev, weights_only=False).items()})
        model.eval()
        with torch.no_grad():
            for b in va:
                img = b["image"].to(dev)
                pri = b["prior"].to(dev)
                gt = b["mask"].numpy()
                variants = [(name, torch.clamp(pri, 0, 1) ** g) for name, g in gammas]
                variants.append(("ikili", MM.binarize_prior(pri)))
                variants.append(("prior yok", torch.zeros_like(pri)))
                for name, p in variants:
                    if dev.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            q = torch.sigmoid(model(img, p).float()).cpu().numpy()
                    else:
                        q = torch.sigmoid(model(img, p).float()).cpu().numpy()
                    for j in range(q.shape[0]):
                        rows.append(dict(pair_id=b["pair_id"][j], fold=k, variant=name,
                                         dice=_dice(q[j, 0], gt[j, 0]),
                                         empty_pred=int((q[j, 0] > 0.5).sum() == 0)))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold {k} bitti ({time.time()-t0:.0f} sn)")

    S = pd.DataFrame(rows)
    if S.empty:
        print("  [-] sonuc yok"); return S

    df, _, _ = dl.load_table()
    keep = [c for c in ("pair_id", "tgt_px", "touch", "cov_soft") if c in df.columns]
    S = S.merge(df[keep], on="pair_id", how="left")

    order = [n for n, _ in gammas] + ["ikili", "prior yok"]
    out = []
    for name in order:
        g = S[S.variant == name]
        row = {"varyant": name, "n": len(g), "Dice": g.dice.mean(),
               "medyan": g.dice.median(), "Dice=0": int((g.dice == 0).sum()),
               "bos tahmin": int(g["empty_pred"].sum())}
        if "touch" in S.columns:
            row["ortusme yok (n=27)"] = g[~g.touch.astype(bool)].dice.mean()
        if "tgt_px" in S.columns:
            row["200-800 px"] = g[(g.tgt_px >= 200) & (g.tgt_px < 800)].dice.mean()
        out.append(row)
    T = pd.DataFrame(out)
    disp = T.copy()
    for c in T.columns:
        if T[c].dtype.kind == "f":
            disp[c] = T[c].map(lambda v: f"{v:.4f}")
    print("\n" + disp.to_string(index=False))
    print(f"""
  Okuma:
    tau=15 satiri makaledeki 0.7384 cift ortalamasini yeniden uretmeli;
    uretmiyorsa boru hattinda sorun vardir (agirlik/fold eslesmesi).
    Egitim araligi 10.7-21.4 mm; bunun disindaki tau'lar dagitim disidir.
    'ikili' satiri g -> sonsuz limitidir ve YUMUSAK prior'la egitilmis bir
    modele ikili girdi vermektir; ikili prior'la EGITILMIS kolun yerini
    tutmaz (asil karsilastirma report()).""")

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        S.to_csv(os.path.join(OUT_DIR, f"tau_sweep_{run}.csv"),
                 index=False, encoding="utf-8-sig")
        T.to_csv(os.path.join(OUT_DIR, f"tau_sweep_{run}_ozet.csv"),
                 index=False, encoding="utf-8-sig")
        print(f"\n  [i] kaydedildi: {OUT_DIR}")
    return S


# ======================================================================
# B) DOGRULAMA (egitimden once)
# ======================================================================
def verify(ckpt=CKPT):
    import torch
    ok = True

    def check(cond, msg_ok, msg_bad):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'[+]' if cond else '[-]'} {msg_ok if cond else msg_bad}")

    hr("1) ON KOSULLAR")
    try:
        import sam2
        print(f"  [+] sam2: {os.path.dirname(sam2.__file__)}")
    except ImportError:
        print("  [-] sam2 kurulu degil"); return False
    if not (os.path.exists(ckpt) and os.path.getsize(ckpt) > 1e8):
        print(f"  [-] kontrol noktasi yok: {ckpt}"); return False
    dl, MM = _modules()
    T = _load("T14", os.path.join(SCRIPTS, "04_train.py"))
    check(hasattr(T, "PRIOR_BINARY"), "04_train.py guncel (PRIOR_BINARY var)",
          "scripts/04_train.py ESKI surum - guncel dosyayi kopyalayin")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2 if dev.type == "cuda" else 1
    print(f"  cihaz: {dev}   batch: {B}")

    # ------------------------------------------------------------------
    hr("2) ESIKLEME FONKSIYONU")
    _, va = dl.build_loaders(0, batch_size=B, augment=False)
    b = next(iter(va))
    pri = b["prior"]
    bin_ = MM.binarize_prior(pri)
    vals = torch.unique(bin_)
    check(set(float(v) for v in vals) <= {0.0, 1.0}, f"cikti {sorted(float(v) for v in vals)}",
          "cikti ikili degil")
    plateau = (pri >= 0.999)
    check(bool((bin_.bool() == plateau).all()),
          "esik, prior platosunu (hizalanmis t-1 maskesi) birebir veriyor",
          "esik plato ile ortusmuyor - esik degerini kontrol edin")
    tail = pri[(pri > 0) & (pri < 0.999)]
    if len(tail):
        check(float(tail.max()) < 0.99,
              f"plato disi en yuksek deger {float(tail.max()):.3f} < 0.99 (esik guvenli)",
              f"plato disi deger {float(tail.max()):.3f} esige cok yakin")
    # gamma augmentasyonu esiklemeden sonra etkisiz mi
    same = all(bool((MM.binarize_prior(torch.clamp(pri, 0, 1) ** g) == bin_).all())
               for g in (0.70, 1.00, 1.40))
    check(same, "egitimdeki prior gamma augmentasyonu esiklemeden sonra etkisiz",
          "gamma augmentasyonu ikili prior'i degistiriyor")
    check(float(MM.binarize_prior(torch.zeros_like(pri)).sum()) == 0.0,
          "prior dropout (sifir prior) ikili kolda da sifir kaliyor",
          "sifir prior esiklemeden sonra sifir degil")

    # ------------------------------------------------------------------
    hr("3) MODEL KURULUMU (ana model ile ayni mimari mi?)")
    t0 = time.time()
    mb = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt, sam2_cfg=CFG,
                        prior_binary=True).to(dev)
    mf = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt, sam2_cfg=CFG).to(dev)
    print(f"  iki model kuruldu ({time.time()-t0:.0f} sn)")
    n_b = sum(v.numel() for v in mb.parameters())
    n_f = sum(v.numel() for v in mf.parameters())
    check(n_b == n_f, f"parametre sayisi ayni ({n_b/1e6:.1f}M)",
          f"parametre sayisi farkli ({n_b} vs {n_f})")
    with torch.no_grad():
        mb.backbone.enc.trunk.patch_embed.proj.weight[:, 3].normal_(0, 0.02)
    miss, unexp = mf.load_state_dict(mb.state_dict(), strict=False)
    check(not miss and list(unexp) == ["prior_binary_tag"],
          "state_dict anahtarlari ayni (tek fark prior_binary_tag)",
          f"anahtar farki: eksik={list(miss)[:3]} fazla={list(unexp)[:3]}")

    # ------------------------------------------------------------------
    hr("4) ESIKLEME FORWARD ICINDE, OMURGA VE GOVDEDEN ONCE MI?")
    img, pri = b["image"].to(dev), b["prior"].to(dev)
    seen = {}
    h1 = mb.backbone.enc.trunk.patch_embed.proj.register_forward_pre_hook(
        lambda m, a: seen.__setitem__("pe", a[0][:, 3:4].detach()))
    h2 = mb.stem.net[0].register_forward_pre_hook(
        lambda m, a: seen.__setitem__("stem", a[0][:, 3:4].detach()))
    mb.eval(); mf.eval()
    with torch.no_grad():
        o_bin = mb(img, pri)
        ref = MM.binarize_prior(pri)
        o_ref = mf(img, ref)
        o_soft = mf(img, pri)
    h1.remove(); h2.remove()
    e_pe = float((seen["pe"] - ref).abs().max())
    e_st = float((seen["stem"] - ref).abs().max())
    check(e_pe < 1e-5, f"patch_embed ikili prior goruyor (fark {e_pe:.1e})",
          f"patch_embed ikili prior GORMUYOR (fark {e_pe:.1e})")
    check(e_st < 1e-5, f"HiResStem ikili prior goruyor (fark {e_st:.1e})",
          f"HiResStem ikili prior GORMUYOR (fark {e_st:.1e})")
    a = float((o_bin - o_ref).abs().max())
    z = float((o_bin - o_soft).abs().max())
    check(a < 1e-3, f"binary(img, prior) == full(img, esiklenmis prior) (fark {a:.1e})",
          f"ciktilar eslesmiyor (fark {a:.1e})")
    check(z > 1e-3, f"esikleme ciktiyi degistiriyor (fark {z:.2e})",
          "yumusak ve ikili prior ayni ciktiyi veriyor - test anlamsiz")

    # ------------------------------------------------------------------
    hr("5) KAYIT KORUMASI")
    try:
        mf.load_state_dict(mb.state_dict()); r1 = False
    except RuntimeError:
        r1 = True
    check(r1, "ikili kol agirliklari ana modele strict yuklenemiyor (dogru)",
          "ikili kol agirliklari ana modele SESSIZCE yuklendi")
    try:
        mb.load_state_dict(mf.state_dict()); r2 = False
    except RuntimeError:
        r2 = True
    check(r2, "ana model agirliklari ikili kola strict yuklenemiyor (dogru)",
          "ana model agirliklari ikili kola SESSIZCE yuklendi")

    # ------------------------------------------------------------------
    hr("6) ILERI + GERI GECIS")
    mb.train()
    crit = MM.ComboLoss()
    out = mb(img, pri)
    loss, log = crit(out.float(), b["mask"].to(dev), epoch=50, total=100)
    loss.backward()
    g_pe = float(mb.backbone.enc.trunk.patch_embed.proj.weight.grad[:, 3].abs().max())
    g_st = float(mb.stem.net[0].weight.grad[:, 3].abs().max())
    print(f"  kayip {loss.item():.4f} (ft={log['ft']:.4f} bd={log['bd']:.4f})")
    check(g_pe > 0, f"patch_embed prior kanali gradyani {g_pe:.2e}", "gradyan yok")
    check(g_st > 0, f"HiResStem prior kanali gradyani {g_st:.2e}", "gradyan yok")
    del mb, mf, out, loss
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    hr("7) 04_train ENTEGRASYONU")
    T.BACKBONE, T.PRIOR_MODE, T.USE_PRIOR, T.USE_STEM = "sam2", "channel", True, True
    T.PRIOR_GRID, T.PRIOR_BINARY, T.RUN_NAME = None, True, None
    tag = T.run_tag()
    check(tag == RUN_BIN, f"kosu adi: {tag}", f"kosu adi beklenmedik: {tag}")
    for run in (RUN_FULL, RUN_PROMPT, RUN_NOPRI):
        n = len(glob.glob(os.path.join(RUNS_DIR, run, "fold*_val_pairs.csv")))
        print(f"  {'[+]' if n == 5 else '[!]'} karsilastirma kosusu {run:<22} {n}/5 fold")

    hr("SONUC")
    if ok:
        print(f"""  Tum kontroller gecti. Egitim:

    T = S.trainer("{RUN_BIN}")      # 13_colab_session.py RUN_CONFIGS'e eklendi
    T.smoke_test(n=10, epochs=30)
    T.run_all()

  Bittiginde: L.report()""")
    else:
        print("  En az bir kontrol basarisiz. Egitime BASLAMAYIN; ciktiyi iletin.")
    return ok


# ======================================================================
# C) RAPOR (egitimden sonra)
# ======================================================================
def report(runs=None, n_boot=N_BOOT, save=True):
    L = _helpers()
    R = dict(soft=RUN_FULL, binary=RUN_BIN, prompt=RUN_PROMPT, noprior=RUN_NOPRI)
    R.update(runs or {})
    NAME = dict(soft="yumusak prior", binary="ikili prior",
                prompt="mask prompt", noprior="prior yok")

    hr("IKILI PRIOR RAPORU")
    loaded = {}
    for k, run in R.items():
        res = L._pairs(run)
        if res is None:
            print(f"  [!] {NAME[k]:<14} {run}: cift dosyasi yok, atlaniyor"); continue
        P, nf = res
        print(f"  {NAME[k]:<14} {run:<24} {nf} fold, {len(P)} cift"
              + ("" if nf == 5 else "   <<< EKSIK FOLD"))
        loaded[k] = P
    if "soft" not in loaded or "binary" not in loaded:
        print("  [-] yumusak ve ikili kollar olmadan rapor uretilemez."); return None
    arms = [k for k in ("soft", "binary", "prompt", "noprior") if k in loaded]

    dl, _ = _modules(need_model=False)
    df, _, _ = dl.load_table()
    pid = "true_patient_id" if "true_patient_id" in df.columns else "patient_key"
    keep = ["pair_id", pid, "tgt_px"] + [c for c in ("touch", "cov_soft") if c in df.columns]
    W = df[keep].rename(columns={pid: "patient"})
    for k in arms:
        W = W.merge(loaded[k].rename(columns={"dice": f"dice_{k}", "hd95": f"hd95_{k}",
                                              "fold": f"fold_{k}"}),
                    on="pair_id", how="inner", validate="one_to_one")
    for k in arms[1:]:
        if not (W[f"fold_{k}"] == W["fold_soft"]).all():
            raise RuntimeError(f"{R[k]} fold atamasi ana modelden farkli")
    W["fold"] = W["fold_soft"]
    W["boyut"] = pd.cut(W.tgt_px, L.SIZE_BINS, labels=L.SIZE_LABELS)
    print(f"  eslesmis cift: {len(W)}   hasta: {W.patient.nunique()}")

    rng = np.random.default_rng(SEED)
    up, inv = np.unique(W.patient.astype(str).values, return_inverse=True)
    C = np.bincount(inv).astype(float)
    IDX = rng.integers(0, len(up), size=(n_boot, len(up)))
    cnt = C[IDX].sum(1)

    def boot_mean(vals, mask=None):
        v = np.asarray(vals, dtype=float)
        if mask is None:
            S = np.bincount(inv, weights=v)
            return S[IDX].sum(1) / cnt
        m = np.asarray(mask, dtype=float)
        S = np.bincount(inv, weights=v * m); N = np.bincount(inv, weights=m)
        n_ = N[IDX].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_ > 0, S[IDX].sum(1) / n_, np.nan)

    def ci(a):
        return np.nanpercentile(a, [2.5, 97.5])

    tables = {}

    # ---------------- 1) genel ----------------
    hr("1) GENEL")
    rows = []
    for k in arms:
        fm = W.groupby("fold")[f"dice_{k}"].mean()
        rows.append({"kol": NAME[k], "n": len(W), "Dice (cift ort.)": f"{W[f'dice_{k}'].mean():.4f}",
                     "fold ort. +/- SD": f"{fm.mean():.4f} +/- {fm.std(ddof=1):.4f}",
                     "medyan": f"{W[f'dice_{k}'].median():.4f}",
                     "Dice=0": int((W[f"dice_{k}"] == 0).sum()),
                     "bos tahmin": int(W[f"hd95_{k}"].isna().sum())})
    t1 = pd.DataFrame(rows); tables["genel"] = t1
    print(t1.to_string(index=False))
    print(f"  [kontrol] yumusak kolun fold ortalamasi makaledeki 0.739 olmali")

    # ---------------- 2) eslesmis karsilastirmalar ----------------
    hr("2) ESLESMIS KARSILASTIRMALAR")
    fam = [(a, b) for a, b in (("binary", "soft"), ("binary", "prompt"), ("binary", "noprior"))
           if a in arms and b in arms]
    rows = []
    for a, b in fam:
        d = W[f"dice_{a}"] - W[f"dice_{b}"]
        lo, hi = ci(boot_mean(W[f"dice_{a}"]) - boot_mean(W[f"dice_{b}"]))
        rows.append(dict(karsilastirma=f"{NAME[a]} - {NAME[b]}", ort=d.mean(),
                         ci_lo=lo, ci_hi=hi, medyan=d.median(),
                         kazanan=f"{(d > 0).sum()}/{len(d)}",
                         p=L._wilcoxon_p(W[f"dice_{a}"], W[f"dice_{b}"])))
    t2 = pd.DataFrame(rows)
    t2["p_holm"] = L._holm(t2["p"].values)
    tables["eslesmis"] = t2
    disp = t2.copy()
    disp["ort (95% GA)"] = [f"{o:+.4f} [{l:+.4f}, {h:+.4f}]"
                            for o, l, h in zip(t2.ort, t2.ci_lo, t2.ci_hi)]
    disp["medyan"] = t2.medyan.map(lambda v: f"{v:+.4f}")
    disp["p"] = t2.p.map(L._fmt_p); disp["p_holm"] = t2.p_holm.map(L._fmt_p)
    print(disp[["karsilastirma", "ort (95% GA)", "medyan", "kazanan", "p", "p_holm"]]
          .to_string(index=False))

    # ---------------- 3) KARAR ALT GRUPLARI ----------------
    hr("3) KARAR ALT GRUPLARI (surekli kodlamanin gerekcesi burada)")
    subs = []
    if "touch" in W.columns:
        subs.append(("ortusme yok (touch=False)", ~W.touch.astype(bool)))
    if "cov_soft" in W.columns:
        q1 = W.cov_soft.quantile(0.25)
        subs.append((f"prior kapsamasi Q1 (<= {q1:.3f})", W.cov_soft <= q1))
    subs.append(("lezyon < 200 px", W.tgt_px < 200))
    subs.append(("lezyon > 800 px", W.tgt_px > 800))
    rows = []
    for label, m in subs:
        m = np.asarray(m, dtype=bool)
        if m.sum() < 5:
            continue
        d = (W[f"dice_binary"] - W[f"dice_soft"])[m]
        lo, hi = ci(boot_mean(W.dice_binary, m) - boot_mean(W.dice_soft, m))
        row = dict(alt_grup=label, n=int(m.sum()),
                   yumusak=W.dice_soft[m].mean(), ikili=W.dice_binary[m].mean(),
                   fark=d.mean(), ci_lo=lo, ci_hi=hi,
                   p=L._wilcoxon_p(W.dice_binary[m], W.dice_soft[m]))
        if "noprior" in arms:
            row["prior yok"] = W.dice_noprior[m].mean()
        rows.append(row)
    t3 = pd.DataFrame(rows)
    t3["p_holm"] = L._holm(t3["p"].values)
    tables["alt_grup"] = t3
    disp = t3.copy()
    for c in ("yumusak", "ikili", "prior yok"):
        if c in disp:
            disp[c] = t3[c].map(lambda v: f"{v:.3f}")
    disp["fark (95% GA)"] = [f"{o:+.3f} [{l:+.3f}, {h:+.3f}]"
                             for o, l, h in zip(t3.fark, t3.ci_lo, t3.ci_hi)]
    disp["p_holm"] = t3.p_holm.map(L._fmt_p)
    cols = ["alt_grup", "n", "yumusak", "ikili"] + (["prior yok"] if "prior yok" in disp else []) \
        + ["fark (95% GA)", "p_holm"]
    print(disp[cols].to_string(index=False))
    print("  Makalenin iddiasi: ortusme olmayan 27 ciftte ikili prior yanlis konumu "
          "kesinlikle bildirir.\n  Bu satirdaki fark ve GA o cumlenin kaderini belirler.")

    # ---------------- 4) boyut tabakalari ----------------
    hr("4) BOYUT TABAKALARI")
    rows = []
    for lab in L.SIZE_LABELS:
        g = W[W.boyut == lab]
        if not len(g):
            continue
        row = {"tabaka": lab, "n": len(g)}
        for k in arms:
            row[NAME[k]] = g[f"dice_{k}"].mean()
        row["fark"] = (g.dice_binary - g.dice_soft).mean()
        row["p"] = L._wilcoxon_p(g.dice_binary, g.dice_soft)
        rows.append(row)
    t4 = pd.DataFrame(rows)
    t4["p_holm"] = L._holm(t4["p"].values)
    tables["tabaka"] = t4
    disp = t4.copy()
    for c in disp.columns:
        if c in NAME.values():
            disp[c] = t4[c].map(lambda v: f"{v:.3f}")
    disp["fark"] = t4.fark.map(lambda v: f"{v:+.3f}")
    disp["p_holm"] = t4.p_holm.map(L._fmt_p)
    print(disp[["tabaka", "n"] + [NAME[k] for k in arms] + ["fark", "p_holm"]]
          .to_string(index=False))

    # ---------------- 5) epoch protokolleri ----------------
    hr("5) EPOCH SECIM PROTOKOLLERI")
    rows = []
    for k in arms:
        f = os.path.join(RUNS_DIR, R[k], "metrics.csv")
        if not os.path.exists(f):
            continue
        mt = (pd.read_csv(f).drop_duplicates(["fold", "epoch"], keep="last")
                .sort_values(["fold", "epoch"]))
        best = mt.groupby("fold").val_dice.max()
        last = mt.groupby("fold").apply(lambda g: g.val_dice.iloc[-1], include_groups=False)
        l10 = mt.groupby("fold").apply(lambda g: g.val_dice.iloc[-10:].mean(),
                                       include_groups=False)
        rows.append({"kol": NAME[k],
                     "epoch/fold": "/".join(str(int(v)) for v in (mt.groupby("fold").epoch.max() + 1)),
                     "en iyi": f"{best.mean():.4f} +/- {best.std(ddof=1):.4f}",
                     "son": f"{last.mean():.4f} +/- {last.std(ddof=1):.4f}",
                     "son 10 ort.": f"{l10.mean():.4f} +/- {l10.std(ddof=1):.4f}"})
    if rows:
        t5 = pd.DataFrame(rows); tables["protokol"] = t5
        print(t5.to_string(index=False))

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        W.to_csv(os.path.join(OUT_DIR, "pairs_binary.csv"), index=False, encoding="utf-8-sig")
        for name, t in tables.items():
            t.to_csv(os.path.join(OUT_DIR, f"rapor_{name}.csv"), index=False,
                     encoding="utf-8-sig")
        print(f"\n  [i] kaydedildi: {OUT_DIR}")

    hr("YORUM REHBERI (dosya basindaki cerceve)")
    print("""  a) ikili ~ yumusak (fark kucuk, GA sifiri kapsiyor, ortusme olmayan alt
     grupta da fark yok): birinci katki ve "ilk kez" cumlesi CIKARILMALI.
     Yumusak kodlama zarar vermiyor ama katki da saglamiyor; yontem
     bolumunde tercih olarak kalir.
  b) ikili < yumusak ve kayip ortusme olmayan / dusuk kapsamali ciftlerde:
     mevcut anlati dogrulanir, iddia bu alt grupla birlikte yazilir.
  c) ikili > yumusak: iddia tersine cevrilir; kuyruk davranisina bakilir.
  Her durumda tau taramasi (tau_sweep) yardimci kanit olarak raporlanir.""")
    return W, tables


if __name__ == "__main__":
    verify()
