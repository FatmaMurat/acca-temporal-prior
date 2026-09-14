# -*- coding: utf-8 -*-
"""
07_sam2_setup.py
----------------
SAM2 kurulumu, agirlik indirme ve ADAPTOR DOGRULAMASI.

SAM2 adaptoru (03_model.py -> SAM2Backbone) su ana kadar hic calistirilmadi;
kaynak kodu okunarak yazildi. Bu script tam kosuya girmeden once adaptoru
uctan uca sinar. 5 fold baslatip 20. dakikada bicim hatasi almak yerine
burada 2 dakikada anlariz.

Kontroller:
  1. sam2 paketi kurulu mu, kontrol noktasi indi mi
  2. ImageEncoder ayaga kalkiyor mu, backbone_fpn kac harita donuyor
  3. 4 kanal adaptasyonu: 4. kanal sifir mi, gradyan akiyor mu
  4. Kanonik piramit bicimleri EVA-02 ile AYNI mi (adil karsilastirma sarti)
  5. Ileri + geri gecis, kayip hesabi
  6. VRAM tepe degeri (batch 2 ve 4)

Kullanim (Colab):
    !pip install -q git+https://github.com/facebookresearch/sam2.git
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("S", "/content/acca/scripts/07_sam2_setup.py")
    S = importlib.util.module_from_spec(s); sys.modules["S"] = S; s.loader.exec_module(S)
    S.setup()      # indirme + dogrulama
"""

import os
import sys
import time
import subprocess

CKPT_DIR  = "/content"
CKPT_NAME = "sam2.1_hiera_base_plus.pt"
CKPT_URL  = ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
             "sam2.1_hiera_base_plus.pt")
CFG       = "configs/sam2.1/sam2.1_hiera_b+.yaml"
MODEL_PY  = "/content/acca/scripts/03_model.py"
IMG_SIZE  = 512


def hr(t=""):
    print("\n" + "=" * 68)
    if t:
        print(t); print("=" * 68)


def ensure_package():
    hr("1) SAM2 PAKETI")
    try:
        import sam2
        print(f"  [+] sam2 kurulu: {os.path.dirname(sam2.__file__)}")
        return True
    except ImportError:
        print("  [-] sam2 kurulu degil. Su komutu calistirin ve tekrar deneyin:")
        print("      !pip install -q git+https://github.com/facebookresearch/sam2.git")
        return False


def ensure_ckpt():
    hr("2) KONTROL NOKTASI")
    p = os.path.join(CKPT_DIR, CKPT_NAME)
    if os.path.exists(p) and os.path.getsize(p) > 1e8:
        print(f"  [+] mevcut: {p}  ({os.path.getsize(p)/1e6:.0f} MB)")
        return p
    print(f"  indiriliyor: {CKPT_URL}")
    r = subprocess.run(["wget", "-q", "--show-progress", "-O", p, CKPT_URL])
    if r.returncode != 0 or os.path.getsize(p) < 1e8:
        print("  [-] indirme basarisiz. Elle deneyin:")
        print(f"      !wget -O {p} {CKPT_URL}")
        return None
    print(f"  [+] indi: {os.path.getsize(p)/1e6:.0f} MB")
    return p


def verify(ckpt):
    import torch
    import importlib.util

    hr("3) ADAPTOR DOGRULAMASI")
    spec = importlib.util.spec_from_file_location("MM", MODEL_PY)
    MM = importlib.util.module_from_spec(spec); sys.modules["MM"] = MM
    spec.loader.exec_module(MM)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  cihaz: {dev}")

    t0 = time.time()
    model = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt, sam2_cfg=CFG)
    print(f"  model kuruldu ({time.time()-t0:.0f} sn)")
    model = model.to(dev)

    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    n_bb  = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    print(f"  parametre: toplam {n_all:.1f}M  (omurga {n_bb:.1f}M)")

    # --- 4 kanal kontrolu ---
    w = model.backbone.enc.trunk.patch_embed.proj.weight
    print(f"  patch_embed girdi kanali : {w.shape[1]}  (4 olmali)")
    print(f"  4. kanal baslangicta sifir: {float(w[:, 3].abs().max()) == 0.0}")

    # --- ileri gecis ---
    img = torch.rand(2, 3, IMG_SIZE, IMG_SIZE, device=dev)
    pri = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, device=dev)
    tgt = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE, device=dev)
    tgt[:, :, 240:280, 240:280] = 1.0

    model.train()
    feats = model.backbone(torch.cat([img, pri], 1))
    print("\n  kanonik piramit:")
    ok = True
    beklenen = [(256, IMG_SIZE // s, IMG_SIZE // s) for s in MM.CANON_STRIDES]
    for f, b in zip(feats, beklenen):
        good = tuple(f.shape[1:]) == b
        ok &= good
        print(f"    {tuple(f.shape)}   beklenen (B, {b[0]}, {b[1]}, {b[2]})"
              f"   {'OK' if good else '<<< UYUSMUYOR'}")
    if ok:
        print("  [+] bicimler EVA-02 ile AYNI -> karsilastirma adil")
    else:
        print("  [-] bicim uyusmazligi! Tam kosuya BASLAMAYIN, bicimleri iletin.")

    out = model(img, pri)
    print(f"\n  model ciktisi: {tuple(out.shape)}")

    crit = MM.ComboLoss()
    loss, log = crit(out.float(), tgt, epoch=50, total=100)
    loss.backward()
    g4 = float(w.grad[:, 3].abs().max())
    print(f"  kayip: {loss.item():.4f}  (ft={log['ft']:.4f} bd={log['bd']:.4f} w={log['w']:.3f})")
    print(f"  4. kanal gradyani akiyor : {g4 > 0}")
    gs = float(model.stem.net[0].weight.grad.abs().max())
    print(f"  govdeye gradyan akiyor   : {gs > 0}")

    # projeksiyon katmanlari optimizer'a gorunuyor mu
    npj = sum(p.numel() for p in model.backbone.proj.parameters())
    print(f"  projeksiyon parametresi  : {npj} "
          f"({'Identity, ek parametre yok' if npj == 0 else 'aktif'})")

    del feats, out, loss
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # --- VRAM ---
    if dev.type == "cuda":
        hr("4) VRAM")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        for bs in (2, 4):
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                x = torch.rand(bs, 3, IMG_SIZE, IMG_SIZE, device=dev)
                q = torch.rand(bs, 1, IMG_SIZE, IMG_SIZE, device=dev)
                y = torch.zeros(bs, 1, IMG_SIZE, IMG_SIZE, device=dev)
                t0 = time.time()
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        l, _ = crit(model(x, q).float(), y, epoch=50, total=100)
                    scaler.scale(l).backward(); scaler.step(opt); scaler.update()
                torch.cuda.synchronize()
                print(f"  batch {bs}: tepe {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, "
                      f"adim {(time.time()-t0)/3*1000:.0f} ms")
                del x, q, y
            except RuntimeError as e:
                msg = "VRAM YETMEDI" if "out of memory" in str(e).lower() else str(e)[:60]
                print(f"  batch {bs}: {msg}")
            finally:
                torch.cuda.empty_cache()

    hr("SONUC")
    if ok:
        print("""  Adaptor calisiyor. Tam kosu icin:

    T.BACKBONE  = "sam2"
    T.SAM2_CKPT = "%s"
    T.RUN_NAME  = None            # -> sam2_prior
    T.USE_PRIOR = True
    T.EPOCHS = 100; T.FREEZE_EPOCHS = 3; T.WARMUP_EP = 5; T.SYNC_EVERY = 5
    T.run_all()""" % os.path.join(CKPT_DIR, CKPT_NAME))
    else:
        print("  Bicim uyusmazligi var; ciktiyi iletin.")
    return ok


def setup():
    if not ensure_package():
        return False
    ckpt = ensure_ckpt()
    if ckpt is None:
        return False
    return verify(ckpt)


if __name__ == "__main__":
    setup()
