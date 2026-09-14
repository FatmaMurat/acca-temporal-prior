# -*- coding: utf-8 -*-
"""
03_model.py
-----------
AC-CA calismasi - Adim 3: model sarmalayicisi ve kayip fonksiyonlari.

TASARIM (adil omurga karsilastirmasi icin):
  Iki omurga da girdiyi 4 KANAL alir (RGB + prior) ve ciktiyi ORTAK bir
  kanonik piramide projelendirir:
        [stride4, stride8, stride16] x 256 kanal  ->  512 girdide 128/64/32
  Ayni decoder, ayni kayip, ayni schedule. Farkli olan tek sey omurga.

  SAM2   : ImageEncoder zaten cok olcekli verir (backbone_fpn, scalp=1
           oldugu icin stride 4/8/16, hepsi 256 kanal). Dogrudan kullanilir.
  EVA-02 : izotropik ViT, tek olcek (518/14 = 37x37, 768 kanal). Uzerine
           ViTDet tarzi basit ozellik piramidi kurulur, sonra kanonik
           boyutlara yeniden orneklenir.

4 KANAL ADAPTASYONU: patch embed conv'u 3->4 genisletilir, 4. kanalin
agirligi SIFIR baslatilir. Boylece egitim tam olarak onceden egitilmis
3-kanal davranisindan baslar; prior'in katkisi ogrenilerek gelir.
(mean-kopya baslangicta ciktiyi bozar ve 194 ciftte toparlanmayabilir.)

NORMALIZASYON: her omurga KENDI on-egitim istatistigini kullanir.
SAM2 ImageNet, EVA-02 CLIP degerleri - ayni degiller, sabit gomulmez,
pretrained_cfg'den okunur. Prior kanali HAM kalir, normalize EDILMEZ.

COZUNURLUK: dataloader 512 verir. EVA-02 patch14 oldugu icin 512 tam
bolunmez (36x14=504, son 8 piksel SESSIZCE dusuyor - olculdu). Bu yuzden
EVA-02 sarmalayicisi iceride 512->518 buyutur, cikisi 518->512 indirir.
SAM2 512'yi dogrudan alir (Hiera toplam adim 32, 512/32=16).

-----------------------------------------------------------------------
SECENEK 3 (prior_mode="prompt")  --  SAM2PromptBackbone
-----------------------------------------------------------------------
Prior 4. kanal yerine SAM2'nin KENDI maske prompt yolundan verilir:

    prior [0,1] (B,1,512,512)
      -> 128x128'e indir  (prompt_enc.mask_input_size)
      -> LOGIT'e cevir    (SAM maske promptu logit bekler, olasilik degil)
      -> sam_prompt_encoder(masks=...)  -> dense (B,256,32,32)
      -> kanonik piramidin stride-16 haritasina EKLENIR

Bu, SAM'in kendi mask_decoder'inin yaptigi islemin ta kendisi
(src = image_embeddings + dense_prompt_embeddings). Fark: biz SAM'in
mask_decoder'ini kullanmiyoruz, ORTAK FPNDecoder'i koruyoruz. Aksi halde
tek deneyde iki sey birden degisir (prompt yolu + decoder) ve ana
karsilastirma konfaunt olur.

UC KRITIK NOKTA:

1) image_size override. Config'de model.image_size=1024, dolayisiyla
   prompt encoder 64x64 gomme ve 256x256 maske girdisi bekler. Bizim
   goruntu kodlayicimiz 512'de 32x32 uretiyor. Bu yuzden build_sam2'ye
   `++model.image_size=512` verilir. Bu override GORUNTU KODLAYICIYI
   DEGISTIRMEZ (Hiera pos_embed'i gercek ozellik boyutuna gore interpole
   eder, config'e bakmaz) -> mevcut sam2_prior kosusu (0.7388) gecerli
   kalir ve karsilastirma eslesmis olur. 09_prompt_verify.py bunu
   deneysel olarak dogrular; korlemesine guvenilmez.

2) 3 KANAL girdi. Prior artik prompt'tan geliyor; 4. kanalda da vermek
   tek degiskenli ablasyonu bozar. HiResStem de 3 kanala iner, yoksa
   prior stride-4'ten sizmaya devam eder.

3) SIFIR BASLATMALI KAPI (gate). FPNDecoder rastgele basliyor; onceden
   egitilmis prompt encoder'in cikti olcegi bu decoder'a gore kalibre
   degil. gate=0 ile egitim tam olarak prior'siz davranistan baslar -
   4. kanalin sifir baslatilmasiyla birebir simetrik. gate=False
   "sadik SAM" varyantidir (duz toplama).
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

CANON_STRIDES = (4, 8, 16)
CANON_CH      = 256
PRIOR_EPS     = 1e-4          # logit donusumunde tasma korumasi


# ===================== 4 kanal patch embed adaptoru ===================
def expand_conv_in_channels(conv, in_ch=4, zero_init_extra=True):
    """Conv2d'yi in_ch girdiye genisletir. Ek kanallar SIFIR baslatilir."""
    if conv.in_channels == in_ch:
        return conv
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, conv.dilation,
                    conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        if zero_init_extra:
            new.weight.zero_()
        else:
            new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1))
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


# ========================= ortak yardimcilar ==========================
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
            nn.GroupNorm(32, cout),      # batch 2-4'te BatchNorm guvenilmez
            nn.GELU())


class SimpleFeaturePyramid(nn.Module):
    """
    ViTDet tarzi: izotropik ViT'in TEK olcekli haritasindan cok olcek uretir.
    stride16 haritasindan: x4 buyutme -> stride4, x2 -> stride8, aynen -> s16.
    """

    def __init__(self, dim, out_ch=CANON_CH):
        super().__init__()
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 2, 2), nn.GroupNorm(32, dim // 2),
            nn.GELU(), nn.ConvTranspose2d(dim // 2, dim // 4, 2, 2))
        self.up2 = nn.ConvTranspose2d(dim, dim // 2, 2, 2)
        self.lat = nn.ModuleList([
            nn.Conv2d(dim // 4, out_ch, 1),
            nn.Conv2d(dim // 2, out_ch, 1),
            nn.Conv2d(dim,      out_ch, 1)])

    def forward(self, x):                      # x: (B, C, h, w) stride ~16
        return [self.lat[0](self.up4(x)),
                self.lat[1](self.up2(x)),
                self.lat[2](x)]


def _to_canonical(feats, img_size):
    """Ozellik haritalarini kanonik boyutlara yeniden ornekler."""
    out = []
    for f, s in zip(feats, CANON_STRIDES):
        t = img_size // s
        if f.shape[-2:] != (t, t):
            f = F.interpolate(f, size=(t, t), mode="bilinear", align_corners=False)
        out.append(f)
    return out


def _normalize_rgb(x, mean, std):
    """Ilk 3 kanali normalize eder; varsa 4. kanal (prior) HAM kalir."""
    rgb = (x[:, :3] - mean) / std
    return rgb if x.shape[1] == 3 else torch.cat([rgb, x[:, 3:]], 1)


# ============================= omurgalar ==============================
class EVA02Backbone(nn.Module):
    """timm EVA-02. Icerde 512->518, cikista kanonik piramit."""

    NAME = "eva02_base_patch14_448"
    NATIVE = 518                                # 14 * 37
    NEEDS_PRIOR = False                         # prior 4. kanaldan gelir

    def __init__(self, img_size=512, pretrained=True, in_ch=4):
        super().__init__()
        import timm
        self.img_size = img_size
        m = timm.create_model(self.NAME, pretrained=pretrained,
                              num_classes=0, img_size=self.NATIVE)
        cfg = m.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        m.patch_embed.proj = expand_conv_in_channels(m.patch_embed.proj, in_ch)
        self.bb = m
        self.dim = m.embed_dim
        self.fpn = SimpleFeaturePyramid(self.dim)

    def normalize(self, x):
        return _normalize_rgb(x, self.mean, self.std)

    def forward(self, x):
        x = self.normalize(x)
        if x.shape[-1] != self.NATIVE:
            x = F.interpolate(x, size=(self.NATIVE,) * 2,
                              mode="bilinear", align_corners=False)
        f = self.bb.forward_features(x)          # (B, N, C)
        B, N, C = f.shape
        g = int(math.isqrt(N))
        f = f[:, N - g * g:, :].transpose(1, 2).reshape(B, C, g, g)
        return _to_canonical(self.fpn(f), self.img_size)

    def set_grad_checkpointing(self, on=True):
        self.bb.set_grad_checkpointing(on)


class SAM2Backbone(nn.Module):
    """
    Gercek SAM2 goruntu kodlayicisi. sam2 paketi ve kontrol noktasi gerekir.

        pip install git+https://github.com/facebookresearch/sam2.git

    ImageEncoder ciktisi: dict(backbone_fpn=[s4, s8, s16], ...) hepsi 256 kanal
    (scalp=1 oldugu icin en dusuk cozunurluk atilir).
    """

    NEEDS_PRIOR = False

    def __init__(self, ckpt, cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
                 img_size=512, in_ch=4, hydra_overrides=None):
        super().__init__()
        from sam2.build_sam import build_sam2
        self.img_size = img_size
        sam = build_sam2(cfg, ckpt, device="cpu",
                         hydra_overrides_extra=list(hydra_overrides or []))
        self.enc = sam.image_encoder
        self._post_build(sam)
        del sam
        # SAM2 ImageNet istatistigi kullanir (sam2/utils/misc.py)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        pe = self.enc.trunk.patch_embed
        pe.proj = expand_conv_in_channels(pe.proj, in_ch)

        # --- projeksiyon katmanlarini SIMDI kur (kuru gecis) ---
        # Tembel kurulum optimizer'dan SONRA parametre yaratirdi; o parametreler
        # hicbir zaman guncellenmez ve kontrol noktasi yuklemesi kirilirdi.
        with torch.no_grad():
            probe = torch.zeros(1, in_ch, img_size, img_size)
            feats = self.enc(self.normalize(probe))["backbone_fpn"][:3]
        self.feat_shapes = [tuple(f.shape[1:]) for f in feats]
        self.proj = nn.ModuleList([
            nn.Identity() if f.shape[1] == CANON_CH else nn.Conv2d(f.shape[1], CANON_CH, 1)
            for f in feats])
        print(f"[SAM2] backbone_fpn cikti bicimleri: {self.feat_shapes}")

    def _post_build(self, sam):
        """Alt siniflar prompt encoder gibi ek parcalari burada tutar."""
        pass

    def normalize(self, x):
        return _normalize_rgb(x, self.mean, self.std)

    def _pyramid(self, x):
        feats = self.enc(self.normalize(x))["backbone_fpn"][:3]
        feats = [p(f) for p, f in zip(self.proj, feats)]
        return _to_canonical(feats, self.img_size)

    def forward(self, x):
        return self._pyramid(x)

    def set_grad_checkpointing(self, on=True):
        warnings.warn("SAM2 tarafinda grad-checkpointing atlandi.")


class SAM2PromptBackbone(SAM2Backbone):
    """
    SECENEK 3: prior SAM2'nin maske prompt yolundan verilir, 4. kanaldan degil.

    Goruntu kodlayici SAM2Backbone ile AYNI (ayni agirlik, ayni normalizasyon,
    ayni kanonik piramit). Tek fark:
      * girdi 3 kanal (patch_embed genisletilmez)
      * prior -> prompt_enc -> dense (B,256,32,32) -> stride-16 haritasina eklenir

    prior=None verilirse SAM2'nin kendi "maske yok" yolu (no_mask_embed)
    kullanilir - bu, prior ablasyonunun DOGRU karsiligi (sifir maske vermek
    degil; sifir maske logit uzayinda -9.2 sabitine denk gelir ve on-egitim
    dagiliminin disindadir).
    """

    NEEDS_PRIOR = True

    def __init__(self, ckpt, cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
                 img_size=512, in_ch=3, gate=True):
        # image_size override SART: prompt encoder gomme boyutunu buradan alir
        super().__init__(ckpt, cfg, img_size, in_ch,
                         hydra_overrides=[f"++model.image_size={img_size}"])
        self.embed_hw = tuple(self.prompt_enc.image_embedding_size)
        self.mask_hw  = tuple(self.prompt_enc.mask_input_size)
        exp = img_size // CANON_STRIDES[-1]
        if self.embed_hw != (exp, exp):
            raise RuntimeError(
                f"prompt encoder gomme boyutu {self.embed_hw}, beklenen ({exp},{exp}). "
                f"image_size override uygulanmamis olabilir.")
        self.use_gate = bool(gate)
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("gate", torch.ones(1))
        print(f"[SAM2-prompt] gomme {self.embed_hw}  maske girdisi {self.mask_hw}  "
              f"gate={'ogrenilir (0 baslangic)' if self.use_gate else 'sabit 1.0'}")

    def _post_build(self, sam):
        self.prompt_enc = sam.sam_prompt_encoder

    # ---------------------------------------------------------------
    def dense_from_prior(self, prior, B, device, dtype):
        """prior [0,1] (B,1,H,W) veya None -> dense gomme (B,256,h,w)."""
        if prior is None:
            d = self.prompt_enc.no_mask_embed.weight.reshape(1, -1, 1, 1)
            return d.expand(B, -1, *self.embed_hw).to(device=device, dtype=dtype)
        m = prior
        if tuple(m.shape[-2:]) != self.mask_hw:
            m = F.interpolate(m, size=self.mask_hw, mode="bilinear",
                              align_corners=False)
        m = m.clamp(PRIOR_EPS, 1.0 - PRIOR_EPS)
        m = torch.log(m / (1.0 - m))             # olasilik -> logit
        _, dense = self.prompt_enc(points=None, boxes=None, masks=m)
        return dense.to(dtype)

    def forward(self, x, prior=None):
        feats = self._pyramid(x)                 # [s4, s8, s16]
        s16 = feats[-1]
        dense = self.dense_from_prior(prior, s16.shape[0], s16.device, s16.dtype)
        feats[-1] = s16 + self.gate.to(s16.dtype) * dense
        return feats


class HiResStem(nn.Module):
    """
    Ham girdiden GERCEK stride-4 ozelligi (512 -> 128, tam cozunurluk).

    NEDEN GEREKLI: EVA-02 patch14 ile 518 girdide 37x37 token uretir, yani
    her hucre ~14 piksel. Olculdu: bu izgarada 200 pikselin altindaki
    lezyonlarin Dice tavani SIFIR, medyan lezyonda (1089 px) ~0.90.
    SimpleFeaturePyramid'in 128x128 haritasi bu bilgiyi yukari orneklemekten
    ibaret - orada detay YOK. Sinir detayini bu govde saglar; ViT anlami
    saglar. Iki omurgada da AYNI, yoksa karsilastirma adil olmaz.

    prior_mode="prompt" iken in_ch=3: prior govdeden de sizmamali.
    """

    def __init__(self, in_ch=4, out_ch=CANON_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),    nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),    nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, out_ch, 1))

    def forward(self, x):
        return self.net(x)


# ============================== decoder ===============================
class FPNDecoder(nn.Module):
    """Iki omurgada da AYNI. Yukaridan asagi birlestirme + segmentasyon kafasi."""

    def __init__(self, ch=CANON_CH, mid=128, img_size=512):
        super().__init__()
        self.img_size = img_size
        self.smooth = nn.ModuleList([ConvBNAct(ch, mid) for _ in CANON_STRIDES])
        self.stem_proj = ConvBNAct(ch, mid)
        self.fuse = nn.Sequential(ConvBNAct(mid, mid), ConvBNAct(mid, mid))
        self.head = nn.Conv2d(mid, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)   # p0 ~ 0.018, pozitif oran %0.64

    def forward(self, feats, stem=None):
        f = [s(x) for s, x in zip(self.smooth, feats)]   # s4, s8, s16
        if stem is not None:
            f[0] = f[0] + self.stem_proj(stem)
        y = f[2]
        for i in (1, 0):
            y = F.interpolate(y, size=f[i].shape[-2:],
                              mode="bilinear", align_corners=False) + f[i]
        y = self.fuse(y)
        y = F.interpolate(y, size=(self.img_size,) * 2,
                          mode="bilinear", align_corners=False)
        return self.head(y)


class PriorGuidedSegmenter(nn.Module):
    """
    prior_mode="channel" : x = [RGB | prior] (4 kanal), omurga + govde 4 kanal
    prior_mode="prompt"  : x = RGB (3 kanal), prior omurgaya AYRI verilir
    """

    def __init__(self, backbone, img_size=512, use_stem=True, prior_mode="channel"):
        super().__init__()
        if prior_mode not in ("channel", "prompt"):
            raise ValueError(f"bilinmeyen prior_mode: {prior_mode}")
        self.prior_mode = prior_mode
        self.in_ch = 4 if prior_mode == "channel" else 3
        self.backbone = backbone
        self.stem = HiResStem(in_ch=self.in_ch) if use_stem else None
        self.decoder = FPNDecoder(img_size=img_size)

    def forward(self, image, prior):
        """image: (B,3,H,W) [0,1]   prior: (B,1,H,W) [0,1] veya None"""
        if self.prior_mode == "channel":
            if prior is None:
                prior = torch.zeros_like(image[:, :1])
            x = torch.cat([image, prior], 1)
            feats = self.backbone(x)
        else:
            x = image
            feats = self.backbone(x, prior)
        return self.decoder(feats, self.stem(x) if self.stem is not None else None)


def build_model(which, img_size=512, pretrained=True, sam2_ckpt=None,
                sam2_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml", use_stem=True,
                prior_mode="channel", prompt_gate=True):
    if prior_mode == "prompt" and which != "sam2":
        raise ValueError("prior_mode='prompt' yalnizca sam2 omurgasinda gecerli.")

    if which == "eva02":
        bb = EVA02Backbone(img_size, pretrained)
    elif which == "sam2":
        if sam2_ckpt is None:
            raise ValueError("sam2 icin sam2_ckpt zorunlu.")
        if prior_mode == "channel":
            bb = SAM2Backbone(sam2_ckpt, sam2_cfg, img_size, in_ch=4)
        else:
            bb = SAM2PromptBackbone(sam2_ckpt, sam2_cfg, img_size,
                                    in_ch=3, gate=prompt_gate)
    else:
        raise ValueError(f"bilinmeyen omurga: {which}")
    return PriorGuidedSegmenter(bb, img_size, use_stem=use_stem,
                                prior_mode=prior_mode)


# ============================== kayiplar ==============================
class FocalTversky(nn.Module):
    """
    alpha < beta -> yanlis negatifi daha cok cezalandirir (recall lehine).
    Lezyon/arka plan orani 155:1 oldugu icin bu yonde ayarli.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
        super().__init__()
        self.a, self.b, self.g, self.eps = alpha, beta, gamma, eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (p * target).sum(dims)
        fp = (p * (1 - target)).sum(dims)
        fn = ((1 - p) * target).sum(dims)
        ti = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        return torch.pow(1 - ti, self.g).mean()


class BoundaryLoss(nn.Module):
    """
    Kervadec ve ark. Hedefin isaretli mesafe haritasi ile tahminin carpimi.
    Mesafe haritasi hedeften tureyip GRADYAN TASIMAZ; egitim sirasinda
    her batch icin hesaplanir (maskeler seyrek, maliyeti dusuk).

    KIRPMA: tamamen bos maskede edt(~m) tum goruntu boyunca buyur (512'de
    ~700'e kadar) ve kayip patlar. +-CLIP ile sinirlanir.
    """

    CLIP = 64.0

    @staticmethod
    @torch.no_grad()
    def signed_distance(target, clip=None):
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt
        clip = BoundaryLoss.CLIP if clip is None else clip
        t = target.detach().cpu().numpy()
        out = np.zeros_like(t, dtype=np.float32)
        for i in range(t.shape[0]):
            m = t[i, 0] > 0.5
            if m.any() and (~m).any():
                out[i, 0] = edt(~m) - edt(m)
            elif not m.any():
                out[i, 0] = edt(~m)
        np.clip(out, -clip, clip, out=out)
        return torch.from_numpy(out).to(target.device)

    def forward(self, logits, target, sdf=None):
        if sdf is None:
            sdf = self.signed_distance(target)
        return (torch.sigmoid(logits) * sdf).mean()


class ComboLoss(nn.Module):
    """
    Focal-Tversky + rampali boundary.
    Boundary sabit agirlikla acilirsa erken egitimde bolgeyi bozar; ilk
    warm_frac orani boyunca kapali, sonra w_max'a dogru dogrusal acilir.
    """

    def __init__(self, w_max=0.5, warm_frac=0.2, **ft):
        super().__init__()
        self.ft = FocalTversky(**ft)
        self.bd = BoundaryLoss()
        self.w_max, self.warm = w_max, warm_frac

    def weight(self, epoch, total):
        f = epoch / max(1, total - 1)
        if f <= self.warm:
            return 0.0
        return self.w_max * (f - self.warm) / max(1e-8, 1 - self.warm)

    def forward(self, logits, target, epoch=0, total=100, sdf=None):
        l_ft = self.ft(logits, target)
        w = self.weight(epoch, total)
        if w == 0.0:
            return l_ft, dict(ft=l_ft.detach().item(), bd=0.0, w=0.0)
        l_bd = self.bd(logits, target, sdf)
        return l_ft + w * l_bd, dict(ft=l_ft.detach().item(),
                                     bd=l_bd.detach().item(), w=w)


# ============================ hizli kontrol ===========================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("EVA-02 yolu (rastgele agirlik, indirme yok):")
    m = build_model("eva02", pretrained=False)
    img = torch.rand(2, 3, 512, 512)
    pri = torch.rand(2, 1, 512, 512)
    tgt = torch.zeros(2, 1, 512, 512); tgt[:, :, 200:240, 200:240] = 1.0

    feats = m.backbone(torch.cat([img, pri], 1))
    for f in feats:
        print(f"   ozellik {tuple(f.shape)}")
    out = m(img, pri)
    print(f"   cikti  {tuple(out.shape)}")

    w = m.backbone.bb.patch_embed.proj.weight
    print(f"   4. kanal baslangicta sifir : {float(w[:, 3].abs().max()) == 0.0}")

    crit = ComboLoss()
    for ep in (0, 25, 99):
        loss, log = crit(out, tgt, epoch=ep, total=100)
        print(f"   epoch {ep:3d}: loss={loss.item():.4f}  ft={log['ft']:.4f} "
              f"bd={log['bd']:.4f}  w={log['w']:.3f}")
    loss.backward()
    print(f"   4. kanal gradyani akiyor   : {float(w.grad[:, 3].abs().max()) > 0}")
    print(f"   parametre: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    print("\nSAM2 prompt yolu icin: 09_prompt_verify.py")
