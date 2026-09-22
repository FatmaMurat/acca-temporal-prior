# -*- coding: utf-8 -*-
"""
12_lowres_prior.py
------------------
AC-CA - Adim 12: prior cozunurluk ablasyonu (on-degerlendirme madde 1.1).

SORU
  Makale 4 kanal yolunun mask-prompt yoluna ustunlugunu "enjeksiyon
  cozunurlugune" bagliyor. Ancak iki yol arasinda cozunurlukten baska
  farklar da var (prior'in girdigi yer, gate, HiResStem). Bu betik yalnizca
  cozunurlugu degistiren kolu dogrular ve analiz eder:

      sam2_prior           : prior 512x512 (ana model)
      sam2_prior_lowres32  : prior 512 -> 32x32 alan ortalamasi -> 512 bilinear
                             (mimari, parametreler, egitim, fold, SEED AYNI)

  32x32 = SAM2 stride-16 izgarasi (prompt yolunun ozellige eklendigi izgara).

YORUM CERCEVESI (sonuclar gorulmeden ONCE yazildi)
  a) lowres32 ~ tam cozunurluk
     4 kanal yolu tam cozunurluklu prior'a ihtiyac duymuyor. Prompt yolunun
     kaybi cozunurlukle degil, prior'in girdigi yer/derinlikle aciklanmali
     (encoder oncesi ve tum FPN seviyeleri vs. yalnizca stride-16).
     Baslik ve iddia yeniden cercevelenir.
  b) lowres32 ~ prompt ya da altinda, kayip 200-800 px bandinda toplaniyor
     Cozunurluk aciklamasiyla tutarli. Tam kanit degil: prompt encoder prior'i
     128x128'de okuyor; buradaki tek kanalli 32x32 ortalama daha siki bir
     darbogaz.
  c) arada
     Kismi katki; boyut tabakalarina gore dagilima bakilir.
  tau = 15 mm (~22 px) zaten yumusak bir harita oldugu icin bozmanin prior'dan
  ne kadar bilgi sildigi egitimden ONCE info_loss() ile olculur.

ADIMLAR (Colab)
  0) Oturum hazirligi: 13_colab_session.prepare(run="sam2_prior_lowres32")
     (Drive'dan veri paketi + guncel scriptler + sam2 + kontrol noktasi).
     Baglanti koptugunda da ayni hucreyle baslanir.
  1) L.info_loss()   CPU, birkac dk. Bozmanin prior'dan ne sildigi. Egitim YOK.
  2) L.verify()      GPU, ~3-5 dk. Bozmanin forward icinde, omurga ve govdeden
                     ONCE yapildigini ve mimarinin ana modelle ayni oldugunu
                     sayisal olarak sinar. Basarisizsa egitime BASLAMAYIN.
  3) Egitim (04_train.py, PRIOR_GRID=32) -> acca_runs/sam2_prior_lowres32
  4) L.report()      CPU. Eslesmis karsilastirma, boyut tabakalari, hasta
                     duzeyinde bootstrap, epoch secim protokolleri.

Kullanim:
    import importlib.util, sys
    def load(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s); sys.modules[name] = m
        s.loader.exec_module(m); return m
    L = load("L", "/content/acca/scripts/12_lowres_prior.py")
    L.info_loss()
    L.verify()

CIKTILAR: acca_runs/_lowres/
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
OUT_DIR    = "/content/drive/MyDrive/acca_runs/_lowres"

CKPT       = "/content/sam2.1_hiera_base_plus.pt"
CFG        = "configs/sam2.1/sam2.1_hiera_b+.yaml"
IMG_SIZE   = 512
GRID       = 32                     # 512 / 32 = 16 px hucre = stride-16

# kosu adlari (04_train.run_tag() ile uretilen adlar)
RUN_FULL   = "sam2_prior"
RUN_LOW    = f"sam2_prior_lowres{GRID}"
RUN_PROMPT = "sam2_prior_prompt"
RUN_NOPRI  = "sam2_noprior"

SIZE_BINS   = [0, 200, 800, 2000, 1e9]          # makale Tablo 4 ile ayni
SIZE_LABELS = ["<200", "200-800", "800-2000", ">2000"]
N_BOOT      = 2000
SEED        = 42
# ======================================================================


def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print(t); print("=" * 72)


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def _modules(need_model=True):
    dl = _load("dl12", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    MM = None
    if need_model:
        MM = _load("MM12", os.path.join(SCRIPTS, "03_model.py"))
        if not hasattr(MM, "degrade_prior"):
            raise RuntimeError("scripts/03_model.py ESKI surum (degrade_prior yok). "
                               "Guncel dosyayi kopyalayin.")
    return dl, MM


# ======================================================================
# 1) BILGI KAYBI  (egitimden once, GPU gerekmez)
# ======================================================================
def info_loss(grid=GRID, save=True):
    """
    244 prior'un her biri icin bozmanin ne sildigini olcer:
      rel_l2     : ||P - D|| / ||P||  (tum goruntu)
      peak_low   : bozulmus prior'un tepe degeri (tam prior'da 1.0)
      p_in, d_in : GUNCEL lezyon maskesi icindeki ortalama prior (tam / bozuk)
      plateau_px : prior'un 1.0 platosu = hizalanmis t-1 maskesi alani (px)
    Boyut tabakasi guncel lezyon alanina (tgt_px) gore.
    """
    import torch
    dl, MM = _modules()
    df, cdir, _ = dl.load_table()
    ds = dl.TemporalPairDataset(df, dl.BASE_DIR, cdir, augment=False, preload=False)

    hr(f"BOZMANIN PRIOR'DAN SILDIGI BILGI  (grid {grid}x{grid}, n = {len(df)})")
    t0, rows, keep = time.time(), [], {}
    for i, r in df.iterrows():
        pid = r["pair_id"]
        p = ds._load_prior(pid).astype(np.float32)
        m = ds._load_mask(r["curr_mask"]) > 0
        d = MM.degrade_prior(torch.from_numpy(p)[None, None], grid)[0, 0].numpy()
        npn = float(np.linalg.norm(p))
        rows.append(dict(
            pair_id=pid, tgt_px=float(r["tgt_px"]),
            plateau_px=int((p >= 0.999).sum()),
            rel_l2=float(np.linalg.norm(p - d) / npn) if npn > 0 else np.nan,
            peak_low=float(d.max()),
            p_in=float(p[m].mean()) if m.any() else np.nan,
            d_in=float(d[m].mean()) if m.any() else np.nan))
        keep[pid] = (p, d, m)
    R = pd.DataFrame(rows)
    R["boyut"] = pd.cut(R.tgt_px, SIZE_BINS, labels=SIZE_LABELS)
    print(f"  hesaplandi ({time.time()-t0:.0f} sn)")

    cell = (IMG_SIZE // grid) ** 2
    def summ(g):
        return pd.Series(dict(
            n=len(g),
            rel_l2_medyan=g.rel_l2.median(),
            tepe_medyan=g.peak_low.median(),
            hedefte_tam=g.p_in.mean(),
            hedefte_bozuk=g.d_in.mean(),
            plato_1hucreden_kucuk=int((g.plateau_px < cell).sum())))
    T = R.groupby("boyut", observed=True).apply(summ, include_groups=False)
    T.loc["TUMU"] = summ(R)
    disp = T.copy()
    for c in ("rel_l2_medyan", "tepe_medyan", "hedefte_tam", "hedefte_bozuk"):
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    disp["n"] = disp["n"].astype(int)
    disp["plato_1hucreden_kucuk"] = disp["plato_1hucreden_kucuk"].astype(int)
    print("\n" + disp.to_string())
    print(f"""
  Okuma:
    rel_l2        : bozmanin sildigi goreli enerji (0 = hic kayip yok)
    tepe          : platonun bozma sonrasi en yuksek degeri (tam prior'da 1.0)
    hedefte_*     : guncel lezyon icindeki ortalama prior - lezyonun
                    bulundugu yere dusen sinyal ne kadar korunuyor
    plato<1 hucre : t-1 maskesi tek hucreden ({cell} px) kucuk olan cift sayisi

  Bu tablo EGITIMDEN ONCE kaydedilmeli. Kayip kucukse ve hedefteki sinyal
  korunuyorsa egitilmis modelin de kazanci buyuk olcude korumasi beklenir
  (senaryo a). Kayip kucuk lezyon tabakalarinda toplaniyorsa (b) icin zemin
  vardir.""")

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        R.to_csv(os.path.join(OUT_DIR, f"info_loss_grid{grid}.csv"),
                 index=False, encoding="utf-8-sig")
        T.to_csv(os.path.join(OUT_DIR, f"info_loss_grid{grid}_ozet.csv"),
                 encoding="utf-8-sig")
        _info_figure(R, keep, grid)
        print(f"\n  [i] kaydedildi: {OUT_DIR}")
    return R, T


def _info_figure(R, keep, grid, win=160):
    """Her tabakadan medyan rel_l2'li cift: tam prior / bozuk prior / fark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = []
    for lab in SIZE_LABELS:
        g = R[(R.boyut == lab) & R.rel_l2.notna()]
        if len(g):
            j = (g.rel_l2 - g.rel_l2.median()).abs().idxmin()
            picks.append((lab, g.loc[j]))
    if not picks:
        return
    c = IMG_SIZE // grid
    fig, ax = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    ax = np.atleast_2d(ax)
    for k, (lab, row) in enumerate(picks):
        p, d, m = keep[row.pair_id]
        ys, xs = np.nonzero(m)
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (IMG_SIZE // 2,) * 2
        y0 = int(np.clip(cy - win // 2, 0, IMG_SIZE - win))
        x0 = int(np.clip(cx - win // 2, 0, IMG_SIZE - win))
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        panels = [(p[sl], "tam prior", 1.0), (d[sl], f"{grid}x{grid} bozuk", 1.0),
                  (np.abs(p - d)[sl], "|fark|", None)]
        for j, (im, title, vmax) in enumerate(panels):
            a = ax[k, j]
            a.imshow(im, cmap="magma", vmin=0, vmax=vmax)
            a.contour(m[sl], levels=[0.5], colors="cyan", linewidths=0.8)
            if j == 1:
                for t in range((-y0) % c, win, c):
                    a.axhline(t - 0.5, color="white", lw=0.3, alpha=0.4)
                for t in range((-x0) % c, win, c):
                    a.axvline(t - 0.5, color="white", lw=0.3, alpha=0.4)
                a.set_xlim(-0.5, win - 0.5); a.set_ylim(win - 0.5, -0.5)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{title}" + (f"\n{lab} px, rel_l2={row.rel_l2:.3f}" if j == 0 else ""),
                        fontsize=8)
    fig.suptitle("Cyan: guncel lezyon (t). Beyaz izgara: 16 px hucreler.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"info_loss_grid{grid}_ornekler.png"), dpi=150)
    plt.close(fig)


# ======================================================================
# 2) DOGRULAMA  (egitimden once, GPU)
# ======================================================================
def verify(ckpt=CKPT, grid=GRID):
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
        print("  [-] sam2 kurulu degil: !pip install -q "
              "git+https://github.com/facebookresearch/sam2.git")
        return False
    if not (os.path.exists(ckpt) and os.path.getsize(ckpt) > 1e8):
        print(f"  [-] kontrol noktasi yok: {ckpt}  (07_sam2_setup.setup())")
        return False
    dl, MM = _modules()
    T = _load("T12", os.path.join(SCRIPTS, "04_train.py"))
    check(hasattr(T, "PRIOR_GRID"), "04_train.py guncel (PRIOR_GRID var)",
          "scripts/04_train.py ESKI surum - guncel dosyayi kopyalayin")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2 if dev.type == "cuda" else 1
    print(f"  cihaz: {dev}   batch: {B}")

    # ------------------------------------------------------------------
    hr("2) BOZMA FONKSIYONU")
    torch.manual_seed(0)
    c = IMG_SIZE // grid
    p = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
    d = MM.degrade_prior(p, grid)
    check(d.shape == p.shape and float(d.min()) >= 0 and float(d.max()) <= 1,
          f"bicim {tuple(d.shape)}, aralik [0,1]", "bicim/aralik bozuk")
    q = (p.view(2, 1, grid, c, grid, c).flip(3).flip(5)
          .reshape(2, 1, IMG_SIZE, IMG_SIZE))            # hucre ICI duzen degisti
    din = float((p - q).abs().max())
    dout = float((MM.degrade_prior(q, grid) - d).abs().max())
    check(din > 0.1 and dout < 1e-5,
          f"hucre ici duzen silinir: girdi farki {din:.2f} -> cikti farki {dout:.1e}",
          f"hucre ici bilgi sizuyor (cikti farki {dout:.1e})")
    check(MM.degrade_prior(p, None) is p, "grid=None -> prior degismez",
          "grid=None prior'u degistiriyor")
    s = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE); s[..., 200:220, 300:320] = 1.0
    sd = MM.degrade_prior(s, grid)
    yy, xx = torch.meshgrid(torch.arange(IMG_SIZE).float(),
                            torch.arange(IMG_SIZE).float(), indexing="ij")
    def cm(t):
        w = t[0, 0]; return (float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum()))
    shift = float(np.hypot(*np.subtract(cm(s), cm(sd))))
    check(shift < 8, f"konum korunur: kutle merkezi kaymasi {shift:.1f} px (< yarim hucre)",
          f"konum kayiyor: {shift:.1f} px")

    # ------------------------------------------------------------------
    hr("3) MODEL KURULUMU (ana model ile ayni mimari mi?)")
    t0 = time.time()
    low = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt, sam2_cfg=CFG,
                         prior_grid=grid).to(dev)
    full = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt, sam2_cfg=CFG,
                          prior_grid=None).to(dev)
    print(f"  iki model kuruldu ({time.time()-t0:.0f} sn)")
    pe = low.backbone.enc.trunk.patch_embed.proj
    check(pe.weight.shape[1] == 4 and pe.stride == (4, 4),
          f"patch_embed 4 kanal, stride {pe.stride}", "patch_embed beklenmedik")
    check(low.stem is not None and low.stem.net[0].weight.shape[1] == 4,
          "HiResStem 4 kanal", "HiResStem 4 kanal degil")
    n_low = sum(v.numel() for v in low.parameters())
    n_full = sum(v.numel() for v in full.parameters())
    check(n_low == n_full, f"parametre sayisi ayni ({n_low/1e6:.1f}M)",
          f"parametre sayisi farkli ({n_low} vs {n_full})")

    # prior'in omurgaya da etki etmesi icin 4. kanal agirliklarini rastgele yap,
    # sonra IKI modeli ayni agirliklara esitle
    with torch.no_grad():
        pe.weight[:, 3].normal_(0, 0.02)
    miss, unexp = full.load_state_dict(low.state_dict(), strict=False)
    check(not miss and list(unexp) == ["prior_grid_tag"],
          "state_dict anahtarlari ayni (tek fark prior_grid_tag)",
          f"anahtar farki: eksik={list(miss)[:3]} fazla={list(unexp)[:3]}")

    # ------------------------------------------------------------------
    hr("4) BOZMA FORWARD ICINDE, OMURGA VE GOVDEDEN ONCE MI?")
    _, va = dl.build_loaders(0, batch_size=B, augment=False)
    b = next(iter(va))
    img, pri = b["image"].to(dev), b["prior"].to(dev)
    print(f"  gercek dogrulama batch'i: {list(b['pair_id'])}")
    seen = {}
    h1 = pe.register_forward_pre_hook(lambda m, a: seen.__setitem__("pe", a[0][:, 3:4].detach()))
    h2 = low.stem.net[0].register_forward_pre_hook(
        lambda m, a: seen.__setitem__("stem", a[0][:, 3:4].detach()))
    low.eval(); full.eval()
    with torch.no_grad():
        o_low = low(img, pri)
        ref = MM.degrade_prior(pri, grid)
        o_ref = full(img, ref)
        o_full = full(img, pri)
    h1.remove(); h2.remove()
    e_pe = float((seen["pe"] - ref).abs().max())
    e_st = float((seen["stem"] - ref).abs().max())
    e_raw = float((seen["pe"] - pri).abs().max())
    check(e_pe < 1e-5, f"patch_embed bozuk prior goruyor (fark {e_pe:.1e})",
          f"patch_embed bozuk prior GORMUYOR (fark {e_pe:.1e})")
    check(e_st < 1e-5, f"HiResStem bozuk prior goruyor (fark {e_st:.1e})",
          f"HiResStem bozuk prior GORMUYOR (fark {e_st:.1e})")
    check(e_raw > 1e-3, f"ham prior ile girdi farkli (fark {e_raw:.2f})",
          "girdi ham prior ile ayni - bozma uygulanmamis")
    a = float((o_low - o_ref).abs().max())
    z = float((o_low - o_full).abs().max())
    check(a < 1e-3, f"low(img, prior) == full(img, bozuk prior)  (fark {a:.1e})",
          f"ciktilar eslesmiyor (fark {a:.1e})")
    check(z > 1e-3, f"cozunurluk ciktiyi degistiriyor (tam vs bozuk fark {z:.2e})",
          "tam ve bozuk prior ayni ciktiyi veriyor - test anlamsiz")

    # ------------------------------------------------------------------
    hr("5) KAYIT KORUMASI")
    try:
        full.load_state_dict(low.state_dict()); r1 = False
    except RuntimeError:
        r1 = True
    check(r1, "lowres agirliklari tam modele strict yuklenemiyor (dogru)",
          "lowres agirliklari tam modele SESSIZCE yuklendi")
    try:
        low.load_state_dict(full.state_dict()); r2 = False
    except RuntimeError:
        r2 = True
    check(r2, "tam agirliklar lowres modele strict yuklenemiyor (dogru)",
          "tam agirliklar lowres modele SESSIZCE yuklendi")
    half = {k: (v.half() if v.is_floating_point() else v) for k, v in low.state_dict().items()}
    check(half["prior_grid_tag"].dtype == torch.long and int(half["prior_grid_tag"]) == grid,
          "fp16 kayitta prior_grid_tag korunuyor", "fp16 kayit etiketi bozuyor")
    low.load_state_dict({k: v.float() for k, v in half.items()})   # 08/10 yukleme kalibi
    check(int(low.prior_grid_tag) == grid, "08/10 yukleme kalibiyla (v.float()) yuklenebiliyor",
          "08/10 yukleme kalibi etiketi bozuyor")

    # ------------------------------------------------------------------
    hr("6) ILERI + GERI GECIS")
    low.train()
    crit = MM.ComboLoss()
    out = low(img, pri)
    loss, log = crit(out.float(), b["mask"].to(dev), epoch=50, total=100)
    loss.backward()
    g_pe = float(pe.weight.grad[:, 3].abs().max())
    g_st = float(low.stem.net[0].weight.grad[:, 3].abs().max())
    print(f"  kayip {loss.item():.4f} (ft={log['ft']:.4f} bd={log['bd']:.4f})")
    check(g_pe > 0, f"patch_embed prior kanali gradyani {g_pe:.2e}", "patch_embed gradyan yok")
    check(g_st > 0, f"HiResStem prior kanali gradyani {g_st:.2e}", "HiResStem gradyan yok")
    del low, full, out, loss
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    hr("7) 04_train ENTEGRASYONU")
    T.BACKBONE, T.PRIOR_MODE, T.USE_PRIOR, T.USE_STEM = "sam2", "channel", True, True
    T.PRIOR_GRID, T.RUN_NAME = grid, None
    tag = T.run_tag()
    check(tag == RUN_LOW, f"kosu adi: {tag}", f"kosu adi beklenmedik: {tag} (beklenen {RUN_LOW})")
    for run in (RUN_FULL, RUN_PROMPT, RUN_NOPRI):
        n = len(glob.glob(os.path.join(RUNS_DIR, run, "fold*_val_pairs.csv")))
        print(f"  {'[+]' if n == 5 else '[!]'} karsilastirma kosusu {run:<22} {n}/5 fold"
              + ("" if n == 5 else "  -> report() icin adi kontrol edin"))
    prog = os.path.join(RUNS_DIR, RUN_LOW, "progress.json")
    if os.path.exists(prog):
        print(f"  [!] {RUN_LOW} klasorunde progress.json VAR: run_all bitmis foldlari atlar.")

    hr("SONUC")
    if ok:
        print(f"""  Tum kontroller gecti. Egitim (ayni oturumda):

    T = load("T", "/content/acca/scripts/04_train.py")
    T.BACKBONE   = "sam2"
    T.SAM2_CKPT  = "{ckpt}"
    T.PRIOR_MODE = "channel"
    T.USE_PRIOR  = True
    T.USE_STEM   = True
    T.PRIOR_GRID = {grid}
    T.RUN_NAME   = None            # -> {RUN_LOW}

    T.smoke_test(n=10, epochs=30)  # birkac dakika; Dice belirgin yukselmeli
    T.run_all()                    # 5 fold

  Oturum koparsa: 13_colab_session.prepare(run="{RUN_LOW}") ile baslayin;
  siradaki hucreyi o yazdirir (T.run_all() kaldigi yerden devam eder).
  Bittiginde: L.report()""")
    else:
        print("  En az bir kontrol basarisiz. Egitime BASLAMAYIN; ciktiyi iletin.")
    return ok


# ======================================================================
# 3) RAPOR  (egitimden sonra, GPU gerekmez)
# ======================================================================
def _pairs(run):
    fs = sorted(glob.glob(os.path.join(RUNS_DIR, run, "fold*_val_pairs.csv")))
    if not fs:
        return None
    out = []
    for f in fs:
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["fold"] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        out.append(d[["pair_id", "fold", "dice", "hd95"]])
    return pd.concat(out, ignore_index=True), len(fs)


def _holm(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok]); vals = p[ok][idx]; n = len(vals)
    adj, run_max = np.empty(n), 0.0
    for i, v in enumerate(vals):
        run_max = max(run_max, (n - i) * v); adj[i] = min(1.0, run_max)
    tmp = np.empty(n); tmp[idx] = adj; out[ok] = tmp
    return out


def _wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if (d != 0).sum() < 6:
        return np.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return np.nan


def _fmt_p(v):
    return "n/a" if pd.isna(v) else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")


def report(runs=None, grid=GRID, n_boot=N_BOOT, save=True):
    """
    runs: kol adlarini degistirmek icin sozluk, or. {"noprior": "sam2_noprior_v2"}
    """
    R = dict(full=RUN_FULL, low=f"sam2_prior_lowres{grid}",
             prompt=RUN_PROMPT, noprior=RUN_NOPRI)
    R.update(runs or {})
    NAME = dict(full="4 kanal (tam)", low=f"4 kanal ({grid}x{grid})",
                prompt="mask prompt", noprior="prior yok")

    hr(f"COZUNURLUK ABLASYONU RAPORU   (grid {grid}x{grid})")
    loaded = {}
    for k, run in R.items():
        res = _pairs(run)
        if res is None:
            print(f"  [!] {NAME[k]:<16} {run}: cift dosyasi yok, bu kol atlaniyor")
            continue
        P, nf = res
        print(f"  {NAME[k]:<16} {run:<24} {nf} fold, {len(P)} cift"
              + ("" if nf == 5 else "   <<< EKSIK FOLD"))
        loaded[k] = P
    if "full" not in loaded or "low" not in loaded:
        print("  [-] tam ve lowres kollari olmadan rapor uretilemez.")
        return None
    arms = [k for k in ("full", "low", "prompt", "noprior") if k in loaded]

    dl, _ = _modules(need_model=False)
    df, _, _ = dl.load_table()
    pid = "true_patient_id" if "true_patient_id" in df.columns else "patient_key"
    W = df[["pair_id", pid, "tgt_px"]].rename(columns={pid: "patient"})
    for k in arms:
        W = W.merge(loaded[k].rename(columns={"dice": f"dice_{k}", "hd95": f"hd95_{k}",
                                              "fold": f"fold_{k}"}),
                    on="pair_id", how="inner", validate="one_to_one")
    if len(W) != len(df):
        print(f"  [!] tum kollarda eslesen cift {len(W)} / {len(df)}")
    for k in arms[1:]:
        if not (W[f"fold_{k}"] == W["fold_full"]).all():
            raise RuntimeError(f"{R[k]} fold atamasi ana modelden farkli - karsilastirma gecersiz")
    W["fold"] = W["fold_full"]
    W["boyut"] = pd.cut(W.tgt_px, SIZE_BINS, labels=SIZE_LABELS)
    print(f"  eslesmis cift: {len(W)}   hasta: {W.patient.nunique()}   (kume: {pid})")

    # ---------------- hasta duzeyinde bootstrap altyapisi ----------------
    rng = np.random.default_rng(SEED)
    pats = W.patient.astype(str).values
    up, inv = np.unique(pats, return_inverse=True)
    C = np.bincount(inv).astype(float)
    S = {k: np.bincount(inv, weights=W[f"dice_{k}"].values) for k in arms}
    IDX = rng.integers(0, len(up), size=(n_boot, len(up)))
    cnt = C[IDX].sum(1)
    BM = {k: S[k][IDX].sum(1) / cnt for k in arms}        # bootstrap ortalamalari
    M0 = {k: W[f"dice_{k}"].mean() for k in arms}

    def ci(arr):
        lo, hi = np.nanpercentile(arr, [2.5, 97.5]); return lo, hi

    tables = {}

    # ---------------- 1) genel ----------------
    hr("1) GENEL (en iyi epoch protokolu - makale ile ayni)")
    rows = []
    for k in arms:
        fm = W.groupby("fold")[f"dice_{k}"].mean()
        rows.append({"kol": NAME[k], "n": len(W),
                     "Dice (cift ort.)": f"{M0[k]:.4f}",
                     "fold ort. +/- SD": f"{fm.mean():.4f} +/- {fm.std(ddof=1):.4f}",
                     "medyan": f"{W[f'dice_{k}'].median():.4f}",
                     "Dice=0": int((W[f"dice_{k}"] == 0).sum()),
                     "bos tahmin": int(W[f"hd95_{k}"].isna().sum())})
    t1 = pd.DataFrame(rows); tables["genel"] = t1
    print(t1.to_string(index=False))
    print("  (bos tahmin = HD95 tanimsiz; hedef maskeler bos olmadigi icin esdeger)")
    print(f"  [kontrol] tam model fold ortalamasi makaledeki 0.739 olmali")
    only_low = int(((W.dice_low == 0) & (W.dice_full > 0)).sum())
    only_full = int(((W.dice_full == 0) & (W.dice_low > 0)).sum())
    print(f"  yalniz lowres'in cokerttigi cift: {only_low}   "
          f"yalniz tam modelin cokerttigi cift: {only_full}")

    # ---------------- 2) eslesmis karsilastirmalar ----------------
    hr("2) ESLESMIS KARSILASTIRMALAR")
    fam = [(a, b) for a, b in (("low", "full"), ("low", "prompt"), ("low", "noprior"))
           if a in arms and b in arms]
    ref = [(a, b) for a, b in (("full", "prompt"), ("full", "noprior"))
           if a in arms and b in arms]
    rows = []
    for a, b in fam + ref:
        d = W[f"dice_{a}"] - W[f"dice_{b}"]
        lo, hi = ci(BM[a] - BM[b])
        rows.append(dict(karsilastirma=f"{NAME[a]} - {NAME[b]}",
                         aile="yeni" if (a, b) in fam else "referans",
                         ort=d.mean(), ci_lo=lo, ci_hi=hi, medyan=d.median(),
                         kazanan=f"{(d > 0).sum()}/{len(d)}",
                         p=_wilcoxon_p(W[f"dice_{a}"], W[f"dice_{b}"])))
    t2 = pd.DataFrame(rows)
    t2["p_holm"] = np.nan
    m = t2.aile == "yeni"
    t2.loc[m, "p_holm"] = _holm(t2.loc[m, "p"].values)
    tables["eslesmis"] = t2
    disp = t2.copy()
    disp["ort (95% GA)"] = [f"{o:+.4f} [{l:+.4f}, {h:+.4f}]"
                            for o, l, h in zip(t2.ort, t2.ci_lo, t2.ci_hi)]
    disp["medyan"] = t2.medyan.map(lambda v: f"{v:+.4f}")
    disp["p"] = t2.p.map(_fmt_p); disp["p_holm"] = t2.p_holm.map(_fmt_p)
    print(disp[["karsilastirma", "aile", "ort (95% GA)", "medyan", "kazanan",
                "p", "p_holm"]].to_string(index=False))
    print("  GA: hasta duzeyinde kume bootstrap (%d tekrar). Holm yalnizca 'yeni' ailede."
          % n_boot)

    # ---------------- 3) kazancin ne kadari korunuyor ----------------
    hr("3) KORUNAN KAZANC")
    if "noprior" in arms:
        den = M0["full"] - M0["noprior"]
        r = (M0["low"] - M0["noprior"]) / den
        rb = (BM["low"] - BM["noprior"]) / (BM["full"] - BM["noprior"])
        lo, hi = ci(rb)
        print(f"  prior kazanci korunumu = (lowres - prior yok) / (tam - prior yok)")
        print(f"      = {r:.2f}   95% GA [{lo:.2f}, {hi:.2f}]   (1 = tam kazanc, 0 = hic)")
        print(f"  NOT: oran GA'lari paydanin bootstrap dagilimina duyarlidir; "
              f"asil cikarim 2. tablodaki farklardan yapilir.")
        tables["korunum_prior"] = dict(oran=r, ci_lo=lo, ci_hi=hi)
    if "prompt" in arms:
        den = M0["full"] - M0["prompt"]
        r = (M0["low"] - M0["prompt"]) / den
        rb = (BM["low"] - BM["prompt"]) / (BM["full"] - BM["prompt"])
        lo, hi = ci(rb)
        print(f"\n  tam-prompt araligindaki konum = (lowres - prompt) / (tam - prompt)")
        print(f"      = {r:.2f}   95% GA [{lo:.2f}, {hi:.2f}]   (1 = tam gibi, 0 = prompt gibi)")
        print(f"  NOT: payda kucuk ({den:+.4f}); GA genis cikarsa tek basina yorumlanmamali.")
        tables["konum_prompt"] = dict(oran=r, ci_lo=lo, ci_hi=hi)

    # ---------------- 4) boyut tabakalari ----------------
    hr("4) BOYUT TABAKALARI (px, 512 izgarasi; Holm 4 tabaka uzerinden)")
    rows = []
    for lab in SIZE_LABELS:
        g = W[W.boyut == lab]
        if not len(g):
            continue
        row = {"tabaka": lab, "n": len(g)}
        for k in arms:
            row[NAME[k]] = g[f"dice_{k}"].mean()
        row["d_low_tam"] = (g.dice_low - g.dice_full).mean()
        row["p_low_tam"] = _wilcoxon_p(g.dice_low, g.dice_full)
        if "prompt" in arms:
            row["d_tam_prompt"] = (g.dice_full - g.dice_prompt).mean()
            row["p_tam_prompt"] = _wilcoxon_p(g.dice_full, g.dice_prompt)
            row["d_low_prompt"] = (g.dice_low - g.dice_prompt).mean()
            row["p_low_prompt"] = _wilcoxon_p(g.dice_low, g.dice_prompt)
        rows.append(row)
    t4 = pd.DataFrame(rows)
    for c in [c for c in t4.columns if c.startswith("p_")]:
        t4[c.replace("p_", "holm_")] = _holm(t4[c].values)
    tables["tabaka"] = t4
    disp = t4.copy()
    for c in disp.columns:
        if c.startswith("d_"):
            disp[c] = disp[c].map(lambda v: f"{v:+.3f}")
        elif c.startswith(("p_", "holm_")):
            disp[c] = disp[c].map(_fmt_p)
        elif c in NAME.values():
            disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    show = ["tabaka", "n"] + [NAME[k] for k in arms] + ["d_low_tam", "holm_low_tam"]
    if "prompt" in arms:
        show += ["d_tam_prompt", "holm_tam_prompt", "d_low_prompt", "holm_low_prompt"]
    print(disp[show].to_string(index=False))
    print("  d_tam_prompt sutunu makale Tablo 4'u yeniden uretmeli "
          "(+0.001 / +0.060 / -0.011 / +0.040).")

    # ---------------- 5) epoch secim protokolleri ----------------
    hr("5) EPOCH SECIM PROTOKOLLERI (fold duzeyi, metrics.csv)")
    rows = []
    for k in arms:
        f = os.path.join(RUNS_DIR, R[k], "metrics.csv")
        if not os.path.exists(f):
            continue
        mt = (pd.read_csv(f).drop_duplicates(["fold", "epoch"], keep="last")
                .sort_values(["fold", "epoch"]))
        n_ep = mt.groupby("fold").epoch.max() + 1
        best = mt.groupby("fold").val_dice.max()
        last = mt.groupby("fold").apply(lambda g: g.val_dice.iloc[-1], include_groups=False)
        last10 = mt.groupby("fold").apply(lambda g: g.val_dice.iloc[-10:].mean(),
                                          include_groups=False)
        rows.append({"kol": NAME[k],
                     "epoch/fold": "/".join(str(int(v)) for v in n_ep.values),
                     "en iyi": f"{best.mean():.4f} +/- {best.std(ddof=1):.4f}",
                     "son": f"{last.mean():.4f} +/- {last.std(ddof=1):.4f}",
                     "son 10 ort.": f"{last10.mean():.4f} +/- {last10.std(ddof=1):.4f}"})
    if rows:
        t5 = pd.DataFrame(rows); tables["protokol"] = t5
        print(t5.to_string(index=False))
        print("  epoch/fold 100 degilse kosu eksik ya da farkli EPOCHS ile kosmus.")

    # ---------------- kayit + yorum ----------------
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        W.to_csv(os.path.join(OUT_DIR, f"pairs_grid{grid}.csv"), index=False,
                 encoding="utf-8-sig")
        for name, t in tables.items():
            (pd.DataFrame([t]) if isinstance(t, dict) else t).to_csv(
                os.path.join(OUT_DIR, f"rapor_grid{grid}_{name}.csv"),
                index=False, encoding="utf-8-sig")
        print(f"\n  [i] kaydedildi: {OUT_DIR}")

    hr("YORUM REHBERI (dosya basindaki cerceve)")
    print("""  a) 'lowres - tam' farki kucuk, GA sifiri kapsiyor ve korunum ~1:
     cozunurluk 4 kanal yolunun ustunlugunu aciklamiyor. Iddia "enjeksiyon
     yeri/derinligi" olarak yeniden kurulmali; baslik degismeli.
  b) lowres prompt duzeyine iniyor ve kayip 200-800 px tabakasinda:
     cozunurluk aciklamasi desteklenir (prompt encoder'in 128x128 okudugu,
     buradaki darbogazin daha siki oldugu not edilerek).
  c) arada: kismi katki. Tabaka tablosunda kaybin nerede toplandigina bakin.
  Her durumda sonuc, one cikmayan yonuyle birlikte raporlanir.""")
    return W, tables


if __name__ == "__main__":
    info_loss()
