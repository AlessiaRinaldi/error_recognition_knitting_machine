#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import numpy as np
import cv2

def preprocess(bgr_or_gray, use_clahe=True):
    # 1) to gray
    if bgr_or_gray.ndim == 3:
        g = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        g = bgr_or_gray.copy()
    # 2) riduzione rumore fine senza ammazzare la trama
    g = cv2.GaussianBlur(g, (3,3), 0)
    # 3) equalizzazione locale per preservare ombre/texture
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        g = clahe.apply(g)
    return g

def load_stack(folder, target_hw=(76,76)):
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None: 
            continue
        # uniforma canali/dimensione
        img = preprocess(img)
        if img.shape[:2] != target_hw:
            img = cv2.resize(img, target_hw[::-1], interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError(f"Nessuna immagine valida in {folder}")
    stack = np.stack(imgs, axis=0)  # [N, H, W]
    return stack, paths

def build_mask(pos_stack, neg_stack, bilateral=True):
    # statistiche per-pixel
    mu_pos = pos_stack.mean(axis=0)
    mu_neg = neg_stack.mean(axis=0)
    std_pos = pos_stack.std(axis=0)
    std_neg = neg_stack.std(axis=0)

    # Fisher-like score (discriminatività per pixel)
    eps = 1e-6
    S = np.abs(mu_pos - mu_neg) / (std_pos + std_neg + eps)

    # Normalizza in [0,1]
    S -= S.min()
    if S.max() > 0:
        S /= S.max()

    # Facoltativo: edge-preserving per non perdere texture
    if bilateral:
        # Bilateral lavora su 8-bit: passa e torna a float
        S8 = (S*255.0).astype(np.uint8)
        S8 = cv2.bilateralFilter(S8, d=5, sigmaColor=15, sigmaSpace=5)
        S = S8.astype(np.float32) / 255.0

    # Hard mask via Otsu
    S8 = (S*255.0).astype(np.uint8)
    thr, hard = cv2.threshold(S8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return S, hard, thr

def save_masks(S, hard, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    soft_path = os.path.join(out_dir, "mask_soft.png")
    hard_path = os.path.join(out_dir, "mask_hard_otsu.png")

    cv2.imwrite(soft_path, (S*255.0).astype(np.uint8))
    cv2.imwrite(hard_path, hard)
    return soft_path, hard_path

def masked_ncc(img, template, mask_float01):
    """
    Esempio: NCC pesata da maschera (valori 0..1).
    img e template devono già essere preprocessati e della stessa size.
    """
    g = img.astype(np.float32)
    t = template.astype(np.float32)
    w = mask_float01.astype(np.float32)

    # centra rispetto a media pesata
    def wmean(x, w):
        return (x*w).sum() / (w.sum() + 1e-6)

    gm = g - wmean(g, w)
    tm = t - wmean(t, w)

    num = (w * gm * tm).sum()
    den = np.sqrt((w * gm*gm).sum()) * np.sqrt((w * tm*tm).sum()) + 1e-6
    return float(num / den)

def main():
    ap = argparse.ArgumentParser(description="Costruisci maschera discriminativa da pos/neg (76x76).")
    ap.add_argument("--pos", required=True, help="Cartella immagini POSITIVE (corrette), es. pos2")
    ap.add_argument("--neg", required=True, help="Cartella immagini NEGATIVE (errate), es. neg2")
    ap.add_argument("--out", default="./mask_out", help="Cartella output per salvataggio maschere")
    ap.add_argument("--no-bilateral", action="store_true", help="Disabilita edge-preserving filter")
    args = ap.parse_args()

    pos_stack, pos_paths = load_stack(args.pos, target_hw=(76,76))
    neg_stack, neg_paths = load_stack(args.neg, target_hw=(76,76))

    S, hard, thr = build_mask(pos_stack, neg_stack, bilateral=(not args.no_bilateral))
    soft_path, hard_path = save_masks(S, hard, args.out)

    print(f"[OK] Soft mask salvata in: {soft_path}")
    print(f"[OK] Hard mask (Otsu) salvata in: {hard_path} (threshold Otsu={thr:.1f})")

    # Esempio d'uso: NCC mascherata con i "prototipi" (medie di classe)
    mu_pos = pos_stack.mean(axis=0)
    mu_neg = neg_stack.mean(axis=0)

    # demo: NCC(img, mu_pos) - NCC(img, mu_neg)
    # prendi una qualsiasi immagine di test dal set neg per esempio
    test_img = neg_stack[0]
    ncc_pos = masked_ncc(test_img, mu_pos, S)  # usa soft mask come pesi
    ncc_neg = masked_ncc(test_img, mu_neg, S)
    llr_like = ncc_pos - ncc_neg
    print(f"Esempio LLR-like su una immagine di test: NCC_pos={ncc_pos:.3f}, NCC_neg={ncc_neg:.3f}, diff={llr_like:.3f}")

if __name__ == "__main__":
    main()
