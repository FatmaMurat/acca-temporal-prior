# -*- coding: utf-8 -*-
"""
11_nnunet_baseline.py
---------------------
AC-CA - Adim 11: nnU-Net v2 taban cizgisi (klasik karsilastirma).

NEDEN GEREKLI
  SAM2 ve EVA-02'nin ikisi de foundation model. Hakem "peki standart bir
  segmentasyon boru hattina karsi ne durumdasiniz" diye sorar. nnU-Net bu
  sorunun kabul gormus cevabidir.

ADIL KARSILASTIRMA ICIN DORT KARAR
  1) GIRDI: 2 kanal (gri goruntu + prior). Bizim modellerimiz RGB+prior alir
     ama RGB, gri BT kesitinin uc kez kopyasidir; bilgi ayni. nnU-Net'e
     prior'siz girdi vermek farki mimariye degil prior'a atfeder ve
     karsilastirmayi cururtur.
  2) FOLD: nnU-Net kendi 5-fold'unu uretir ve bu HASTA SEVIYESI DEGILDIR
     -> sizinti. splits_final.json bizim bolunmemizle EZILIR.
  3) VERI: dataloader'dan augment=False ile disa aktarilir. nnU-Net tam olarak
     bizim modelimizin gordugu 512'lik diziyi alir. Yeniden orneklem yok,
     spacing 1 (goruntuler zaten cift bazinda ortak izgaraya getirilmistir).
  4) DEGERLENDIRME: nnU-Net'in kendi metrikleri DEGIL, bizim dice_iou_np /
     hd95 fonksiyonlarimiz kullanilir. Cikti fold{k}_val_pairs.csv olarak
     yazilir; boylece 06_ablation_compare.py dogrudan calisir.

SURE
  nnUNetTrainer_100epochs: fold basina ~2-2.5 sa (T4) -> 5 fold ~12 sa.
  nnUNetTrainer_50epochs : ~6 sa. 100 epoch = 25000 iterasyon; 195 goruntuluk
  egitim kumesinde ~1500 gecis, bu veri boyutunda fazlasiyla yeterli.
  Kisaltma yontemde ACIKCA belirtilmelidir.

KULLANIM (Colab)
    !pip install -q nnunetv2
    N = load("N", "/content/acca/scripts/11_nnunet_baseline.py")
    N.prepare()                 # ~2 dk, veri + splits
    N.plan_and_preprocess()     # ~3 dk
    print(N.train_commands())   # bu komutlari sirayla calistirin
    # ... egitim bittikten sonra:
    N.predict_all()
    N.evaluate()                # -> acca_runs/nnunet_prior/fold*_val_pairs.csv
"""

import os
import sys
import json
import shutil
import importlib.util

import numpy as np
import pandas as pd
import cv2

# ------------------------------- yollar -------------------------------
BASE_DIR   = "/content/acca"
CACHE_ROOT = "/content/acca/_prior_cache"
SCRIPTS    = "/content/acca/scripts"
RUNS_DIR   = "/content/drive/MyDrive/acca_runs"

NN_ROOT    = "/content/nnunet"
RAW        = f"{NN_ROOT}/nnUNet_raw"
PREP       = f"{NN_ROOT}/nnUNet_preprocessed"
RES        = f"{NN_ROOT}/nnUNet_results"

DATASET_ID   = 501
DATASET_NAME = f"Dataset{DATASET_ID}_ACCA"
TRAINER      = "nnUNetTrainer_100epochs"   # 50epochs ile yariya iner
CONFIG       = "2d"
USE_PRIOR    = True                        # False -> 1 kanal (standart nnU-Net)
RUN_NAME     = "nnunet_prior"              # -> acca_runs/<RUN_NAME>


def _setenv():
    os.environ["nnUNet_raw"] = RAW
    os.environ["nnUNet_preprocessed"] = PREP
    os.environ["nnUNet_results"] = RES
    for p in (RAW, PREP, RES):
        os.makedirs(p, exist_ok=True)


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def _imwrite_u(path, arr):
    """Turkce yol kurali: cv2.imwrite YOK."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(path)
    buf.tofile(path)


# ===================== 1) veriyi nnU-Net formatina cevir =====================
def prepare(folds=(0, 1, 2, 3, 4)):
    """
    dataloader'dan augment=False ile okur, nnU-Net raw formatina yazar.
      imagesTr/<case>_0000.png  gri goruntu (uint8)
      imagesTr/<case>_0001.png  prior      (uint8, USE_PRIOR ise)
      labelsTr/<case>.png       maske      (uint8 {0,1})
    Ayrica splits_final.json BIZIM foldlarimizla yazilir.
    """
    _setenv()
    dl = _load("dlN", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)

    droot = os.path.join(RAW, DATASET_NAME)
    itr, ltr = os.path.join(droot, "imagesTr"), os.path.join(droot, "labelsTr")
    shutil.rmtree(droot, ignore_errors=True)
    os.makedirs(itr); os.makedirs(ltr)

    splits, n = [], 0
    for k in folds:
        tr, va = dl.build_loaders(k, batch_size=1, augment=False)
        fold_ids = {"train": [], "val": []}
        for split, loader in (("train", tr), ("val", va)):
            ds = loader.dataset
            for i in range(len(ds)):
                s = ds[i]
                cid = str(s["pair_id"])
                fold_ids[split].append(cid)
                if split == "val" or not os.path.exists(
                        os.path.join(ltr, cid + ".png")):
                    img = (s["image"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
                    _imwrite_u(os.path.join(itr, f"{cid}_0000.png"), img)
                    if USE_PRIOR:
                        pri = (s["prior"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
                        _imwrite_u(os.path.join(itr, f"{cid}_0001.png"), pri)
                    msk = (s["mask"][0].numpy() > 0.5).astype(np.uint8)
                    _imwrite_u(os.path.join(ltr, f"{cid}.png"), msk)
                    n += 1
        splits.append(fold_ids)
        print(f"  fold {k}: train={len(fold_ids['train'])} val={len(fold_ids['val'])}")

    ch = {"0": "CT_slice"} | ({"1": "prior"} if USE_PRIOR else {})
    with open(os.path.join(droot, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump({
            "channel_names": ch,
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(set(sum([s["train"] + s["val"] for s in splits], []))),
            "file_ending": ".png",
        }, f, indent=2)

    os.makedirs(os.path.join(PREP, DATASET_NAME), exist_ok=True)
    sp = os.path.join(PREP, DATASET_NAME, "splits_final.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\n  yazilan benzersiz vaka : {n}")
    print(f"  kanal sayisi           : {len(ch)}  {list(ch.values())}")
    print(f"  splits (BIZIM foldlar) : {sp}")
    # sizinti kontrolu
    for k, s in enumerate(splits):
        assert not (set(s["train"]) & set(s["val"])), f"fold {k} SIZINTI"
    print("  [+] train/val kesisimi yok")
    return droot


def plan_and_preprocess():
    _setenv()
    cmd = (f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} "
           f"-c {CONFIG} --verify_dataset_integrity")
    print(cmd)
    os.system(cmd)
    # splits_final.json'un ezilmedigini dogrula
    sp = os.path.join(PREP, DATASET_NAME, "splits_final.json")
    s = json.load(open(sp, encoding="utf-8"))
    print(f"  splits_final.json: {len(s)} fold, fold0 val={len(s[0]['val'])} "
          f"{'[+] bizim bolunmemiz duruyor' if len(s) == 5 else '[-] KONTROL EDIN'}")


def train_commands(folds=(0, 1, 2, 3, 4)):
    _setenv()
    L = [f"export nnUNet_raw={RAW}", f"export nnUNet_preprocessed={PREP}",
         f"export nnUNet_results={RES}"]
    for k in folds:
        L.append(f"nnUNetv2_train {DATASET_ID} {CONFIG} {k} -tr {TRAINER} --npz")
    return "\n".join(L)


# ============================ 2) cikarim =============================
def predict_all(folds=(0, 1, 2, 3, 4)):
    """Her fold, KENDI val vakalari uzerinde tahmin eder."""
    _setenv()
    droot = os.path.join(RAW, DATASET_NAME)
    splits = json.load(open(os.path.join(PREP, DATASET_NAME,
                                         "splits_final.json"), encoding="utf-8"))
    for k in folds:
        ind = f"{NN_ROOT}/infer/fold{k}/in"
        out = f"{NN_ROOT}/infer/fold{k}/out"
        shutil.rmtree(f"{NN_ROOT}/infer/fold{k}", ignore_errors=True)
        os.makedirs(ind); os.makedirs(out)
        for cid in splits[k]["val"]:
            for c in (["_0000", "_0001"] if USE_PRIOR else ["_0000"]):
                shutil.copy2(os.path.join(droot, "imagesTr", f"{cid}{c}.png"),
                             os.path.join(ind, f"{cid}{c}.png"))
        cmd = (f"nnUNetv2_predict -i {ind} -o {out} -d {DATASET_ID} "
               f"-c {CONFIG} -f {k} -tr {TRAINER}")
        print(f"\n[fold {k}] {len(splits[k]['val'])} vaka\n{cmd}")
        os.system(cmd)


# ========================= 3) bizim metriklerimiz =========================
def evaluate(folds=(0, 1, 2, 3, 4), run_name=None):
    """nnU-Net'in kendi degerlendirmesi DEGIL; 04_train.py'deki fonksiyonlar."""
    _setenv()
    T = _load("TN", os.path.join(SCRIPTS, "04_train.py"))
    dl = _load("dlN2", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)

    run = run_name or RUN_NAME
    rd = os.path.join(RUNS_DIR, run)
    os.makedirs(rd, exist_ok=True)
    ozet = []

    for k in folds:
        _, va = dl.build_loaders(k, batch_size=1, augment=False)
        df = va.dataset.df.set_index("pair_id")
        out = f"{NN_ROOT}/infer/fold{k}/out"
        rows = []
        for cid in df.index:
            p = os.path.join(out, f"{cid}.png")
            if not os.path.exists(p):
                print(f"  [eksik] {p}"); continue
            pred = dl.robust_read_mask(p) > 0.5
            gt = dl.robust_read_mask(os.path.join(
                RAW, DATASET_NAME, "labelsTr", f"{cid}.png")) > 0.5
            d, iou = T.dice_iou_np(pred.astype(float), gt.astype(float))
            rows.append(dict(pair_id=cid, dice=d, iou=iou,
                             hd95=T.hd95(pred, gt),
                             dice_copy=float(df.loc[cid, "dice_copy_aligned"])))
        r = pd.DataFrame(rows)
        r.to_csv(os.path.join(rd, f"fold{k}_val_pairs.csv"),
                 index=False, encoding="utf-8-sig")
        ozet.append((k, r.dice.mean(), r.dice_copy.mean(), int(r.hd95.isna().sum())))
        print(f"  fold {k}: Dice {r.dice.mean():.4f}  taban {r.dice_copy.mean():.4f}  "
              f"HD95 {r.hd95.mean():.1f}  (tanimsiz {int(r.hd95.isna().sum())})")

    b = np.array([x[1] for x in ozet])
    print(f"\n  ORTALAMA Dice {b.mean():.4f} +/- {b.std(ddof=1):.4f}")
    print(f"  cikti: {rd}")
    print(f"\n  karsilastirma icin:\n    C.fold_table('sam2_prior', '{run}')\n"
          f"    M = C.ablation('sam2_prior', '{run}')")
    return ozet
