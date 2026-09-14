# -*- coding: utf-8 -*-
"""
09_prompt_verify.py
-------------------
SECENEK 3 (SAM2 maske prompt yolu) adaptorunun uctan uca dogrulanmasi.

07_sam2_setup.py'nin mantigini izler: 5 fold baslatip 20. dakikada bicim
hatasi almak yerine burada 3 dakikada anlariz. Ama burada bir dogrulama
DAHA var ve makale acisindan kritik:

  ++model.image_size=512 override'i GORUNTU KODLAYICIYI DEGISTIRIYOR MU?

Degistiriyorsa mevcut sam2_prior kosusu (Dice 0.7388) ile secenek 3
eslesmis bir karsilastirma OLMAZ ve 6 saat bosa gider. Kontrol 5 bunu
sayisal olarak sinar: iki omurga ayni RGB girdisinde BIREBIR ayni
backbone_fpn haritalarini uretmelidir.
(4 kanal omurgada 4. kanal agirligi sifir oldugu icin prior kanali
ciktiyi etkilemez - ayni kontrol sifir baslatmayi da dogrular.)

Kontroller:
  1. sam2 paketi + kontrol noktasi
  2. prompt encoder gomme/maske boyutlari (32x32 / 128x128 olmali)
  3. dense gomme bicimi kanonik stride-16 ile ayni mi
  4. gate=0 iken model ciktisi prior'dan BAGIMSIZ mi (simetri kontrolu)
  5. image_size override goruntu kodlayiciyi degistirmiyor mu  <<< kritik
  6. ileri + geri gecis; prompt_enc ve gate'e gradyan akiyor mu
  7. optimizer gruplamasi: prompt_enc omurga grubunda mi
  8. VRAM tepe degeri (batch 2 ve 4)

Kullanim (Colab):
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("V", "/content/acca/scripts/09_prompt_verify.py")
    V = importlib.util.module_from_spec(s); sys.modules["V"] = V; s.loader.exec_module(V)
    V.verify()
"""

import os
import sys
import time
import importlib.util

CKPT     = "/content/sam2.1_hiera_base_plus.pt"
CFG      = "configs/sam2.1/sam2.1_hiera_b+.yaml"
MODEL_PY = "/content/acca/scripts/03_model.py"
TRAIN_PY = "/content/acca/scripts/04_train.py"
IMG_SIZE = 512


def hr(t=""):
    print("\n" + "=" * 68)
    if t:
        print(t); print("=" * 68)


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def verify(ckpt=CKPT, compare_encoders=True):
    import torch

    hr("1) ON KOSULLAR")
    try:
        import sam2
        print(f"  [+] sam2: {os.path.dirname(sam2.__file__)}")
    except ImportError:
        print("  [-] sam2 kurulu degil:")
        print("      !pip install -q git+https://github.com/facebookresearch/sam2.git")
        return False
    if not (os.path.exists(ckpt) and os.path.getsize(ckpt) > 1e8):
        print(f"  [-] kontrol noktasi yok: {ckpt}  (07_sam2_setup.setup() calistirin)")
        return False
    print(f"  [+] ckpt: {ckpt} ({os.path.getsize(ckpt)/1e6:.0f} MB)")

    MM = _load("MM", MODEL_PY)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  cihaz: {dev}")

    hr("2) PROMPT OMURGASI KURULUMU")
    t0 = time.time()
    model = MM.build_model("sam2", img_size=IMG_SIZE, sam2_ckpt=ckpt,
                           sam2_cfg=CFG, prior_mode="prompt", prompt_gate=True)
    print(f"  kuruldu ({time.time()-t0:.0f} sn)")
    model = model.to(dev).train()
    bb = model.backbone

    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    n_pe  = sum(p.numel() for p in bb.prompt_enc.parameters()) / 1e6
    print(f"  parametre: toplam {n_all:.1f}M  (prompt_enc {n_pe:.2f}M)")

    exp = IMG_SIZE // MM.CANON_STRIDES[-1]
    ok = True
    g1 = bb.embed_hw == (exp, exp)
    g2 = bb.mask_hw == (4 * exp, 4 * exp)
    ok &= g1 and g2
    print(f"  gomme boyutu     : {bb.embed_hw}  beklenen ({exp},{exp})   "
          f"{'OK' if g1 else '<<< UYUSMUYOR'}")
    print(f"  maske girdi boyutu: {bb.mask_hw}  beklenen ({4*exp},{4*exp}) "
          f"{'OK' if g2 else '<<< UYUSMUYOR'}")
    pe_in = bb.enc.trunk.patch_embed.proj.weight.shape[1]
    g3 = pe_in == 3
    ok &= g3
    print(f"  patch_embed girdi : {pe_in}  beklenen 3 (prior kanaldan GELMEZ) "
          f"{'OK' if g3 else '<<< UYUSMUYOR'}")
    stem_in = model.stem.net[0].weight.shape[1] if model.stem is not None else None
    print(f"  HiResStem girdi   : {stem_in}  beklenen 3 "
          f"{'OK' if stem_in == 3 else '<<< prior govdeden siziyor'}")
    ok &= (stem_in == 3)

    hr("3) KANONIK PIRAMIT + DENSE GOMME")
    img = torch.rand(2, 3, IMG_SIZE, IMG_SIZE, device=dev)
    pri = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, device=dev)
    tgt = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE, device=dev)
    tgt[:, :, 240:280, 240:280] = 1.0

    feats = bb(img, pri)
    beklenen = [(MM.CANON_CH, IMG_SIZE // s, IMG_SIZE // s) for s in MM.CANON_STRIDES]
    for f, b in zip(feats, beklenen):
        good = tuple(f.shape[1:]) == b
        ok &= good
        print(f"    {tuple(f.shape)}   beklenen (B, {b[0]}, {b[1]}, {b[2]})"
              f"   {'OK' if good else '<<< UYUSMUYOR'}")
    with torch.no_grad():
        dense = bb.dense_from_prior(pri, 2, dev, feats[-1].dtype)
        dnone = bb.dense_from_prior(None, 2, dev, feats[-1].dtype)
    print(f"  dense(prior) {tuple(dense.shape)}   dense(None) {tuple(dnone.shape)}")
    gd = dense.shape[1:] == feats[-1].shape[1:] == dnone.shape[1:]
    ok &= gd
    print(f"  stride-16 ile toplanabilir: {'OK' if gd else '<<< UYUSMUYOR'}")
    print(f"  prior/None dense farkli   : "
          f"{float((dense-dnone).abs().max()) > 1e-4}  (farkli olmali)")

    hr("4) SIFIR KAPI SIMETRISI  (gate=0 -> prior etkisiz)")
    model.eval()
    with torch.no_grad():
        a = model(img, pri)
        b_ = model(img, None)
    dmax = float((a - b_).abs().max())
    g4 = dmax < 1e-4
    ok &= g4
    print(f"  |model(prior) - model(None)| max = {dmax:.2e}   "
          f"{'OK - egitim prior`siz davranistan basliyor' if g4 else '<<< KAPI SIFIR DEGIL'}")
    model.train()

    if compare_encoders:
        hr("5) image_size OVERRIDE GORUNTU KODLAYICIYI DEGISTIRIYOR MU?  <<< KRITIK")
        print("  4 kanalli (override'siz) omurga kuruluyor...")
        ch = MM.SAM2Backbone(ckpt, CFG, IMG_SIZE, in_ch=4).to(dev).eval()
        x4 = torch.cat([img, pri], 1)
        with torch.no_grad():
            f_ch = ch._pyramid(x4)
            f_pr = bb._pyramid(img)
        worst = 0.0
        for i, (u, v) in enumerate(zip(f_ch, f_pr)):
            d = float((u - v).abs().max()); worst = max(worst, d)
            print(f"    stride{MM.CANON_STRIDES[i]:>2}: max fark {d:.3e}")
        g5 = worst < 1e-3
        ok &= g5
        if g5:
            print("  [+] goruntu kodlayici AYNI -> sam2_prior (0.7388) ile")
            print("      eslesmis karsilastirma gecerli. Ayrica 4. kanalin sifir")
            print("      baslatildigi da dogrulanmis oldu.")
        else:
            print("  [-] KODLAYICI FARKLI. Secenek 3'u BASLATMAYIN; override")
            print("      goruntu kodlayiciyi etkiliyor, karsilastirma eslesmez.")
        del ch, f_ch, f_pr
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    hr("6) ILERI + GERI GECIS")
    crit = MM.ComboLoss()
    out = model(img, pri)
    print(f"  cikti {tuple(out.shape)}")
    loss, log = crit(out.float(), tgt, epoch=50, total=100)
    loss.backward()
    gpe = max(float(p.grad.abs().max()) for p in bb.prompt_enc.parameters()
              if p.grad is not None)
    gga = float(bb.gate.grad.abs().max())
    gst = float(model.stem.net[0].weight.grad.abs().max())
    print(f"  kayip {loss.item():.4f} (ft={log['ft']:.4f} bd={log['bd']:.4f} w={log['w']:.3f})")
    print(f"  prompt_enc gradyani : {gpe:.3e}  {'OK' if gpe > 0 else '<<< AKMIYOR'}")
    print(f"  gate gradyani       : {gga:.3e}  {'OK' if gga > 0 else '<<< AKMIYOR'}")
    print(f"  HiResStem gradyani  : {gst:.3e}  {'OK' if gst > 0 else '<<< AKMIYOR'}")
    ok &= (gpe > 0 and gga > 0)
    model.zero_grad(set_to_none=True)

    hr("7) OPTIMIZER GRUPLAMASI")
    try:
        T = _load("T", TRAIN_PY)
        bb_n = [n for n, p in model.named_parameters()
                if n.startswith(T._BB_PREFIXES)]
        hd_n = [n for n, p in model.named_parameters()
                if not n.startswith(T._BB_PREFIXES)]
        n_pe_bb = sum(1 for n in bb_n if n.startswith("backbone.prompt_enc"))
        n_pe_hd = sum(1 for n in hd_n if n.startswith("backbone.prompt_enc"))
        print(f"  omurga grubu {len(bb_n)} tensor  (prompt_enc: {n_pe_bb})")
        print(f"  kafa grubu   {len(hd_n)} tensor  (prompt_enc: {n_pe_hd})")
        print(f"  gate kafa grubunda: {'backbone.gate' in hd_n} (dogrusu True)")
        g7 = n_pe_bb > 0 and n_pe_hd == 0
        ok &= g7
        print("  " + ("[+] prompt_enc LR_BACKBONE ile guncellenecek"
                      if g7 else
                      "[-] prompt_enc KAFA grubunda! 5e-4 + WD 0.05 on-egitimi bozar."))
    except Exception as e:
        print(f"  [not] 04_train.py yuklenemedi ({type(e).__name__}: {e}) - atlandi")

    if dev.type == "cuda":
        hr("8) VRAM")
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

    T.BACKBONE    = "sam2"
    T.SAM2_CKPT   = "%s"
    T.PRIOR_MODE  = "prompt"
    T.PROMPT_GATE = True
    T.USE_PRIOR   = True
    T.USE_STEM    = True
    T.RUN_NAME    = None          # -> sam2_prior_prompt
    T.EPOCHS = 100; T.FREEZE_EPOCHS = 3; T.WARMUP_EP = 5; T.SYNC_EVERY = 5

    T.smoke_test(n=10, epochs=30)   # ONCE bu: Dice 0.85 ustune cikmali
    T.run_all()                     # sonra bu

  Not: 06_ablation_compare.py ile eslesmis karsilastirma
       sam2_prior  vs  sam2_prior_prompt  (ayni fold, ayni SEED).""" % ckpt)
    else:
        print("  En az bir kontrol basarisiz. Tam kosuya BASLAMAYIN; ciktiyi iletin.")
    return ok


if __name__ == "__main__":
    verify()
