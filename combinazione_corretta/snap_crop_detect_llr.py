#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snap_crop_detect_llr.py
-----------------------
Scatta una foto ogni volta che il microinterruttore viene premuto, ritaglia la
stessa ROI già usata nel tuo script, ed esegue il riconoscimento LLR in tempo reale
stampando a terminale "ok" oppure "error".

Dipendenze (Raspberry Pi OS):
  sudo apt update
  sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-numpy

Esempio d'uso:
  python3 snap_crop_detect_llr.py \
      --pos ./pos \
      --neg ./neg \
      --save-dir /home/aless/foto \
      --thr 0.10

Opzionali:
  --roi-mode relative|pixels    (default: relative)
  --roi-rel 0.47 0.75 0.06 0.08 (rx ry rw rh, se relative)
  --roi-pix 560 760 120 120     (x y w h, se pixels)
  --annotate                    (salva crop con etichetta OK/ERROR)
  --roi-mask path.png           (mask 8-bit per rifinitura dentro al ritaglio)
"""
import os, time, glob, argparse
from datetime import datetime
from gpiozero import Button
from picamera2 import Picamera2
import cv2
import numpy as np

# ---------------------- Utility comuni ----------------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def preprocess(bgr_or_gray):
    g = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY) if bgr_or_gray.ndim==3 else bgr_or_gray
    g = cv2.GaussianBlur(g, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    return clahe.apply(g)

def ncc(A, B, mask=None):
    A = A.astype(np.float32); B = B.astype(np.float32)
    if mask is not None:
        m = (mask>0).astype(np.float32)
        if m.sum() < 10: return 0.0
        A = A*m; B = B*m
        muA = A.sum()/m.sum(); muB = B.sum()/m.sum()
        A = (A - muA)*m; B = (B - muB)*m
        denom = (np.sqrt((A*A).sum()) * np.sqrt((B*B).sum()) + 1e-8)
        return float((A*B).sum() / denom)
    else:
        A = A - A.mean(); B = B - B.mean()
        denom = (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)
        return float((A*B).sum() / denom)

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x)); h = max(1, min(int(h), H - y))
    return x, y, w, h

def roi_from_mode(W, H, mode, roi_rel, roi_pix):
    if mode == "relative":
        rx, ry, rw, rh = roi_rel
        x = int(rx * W); y = int(ry * H); w = int(rw * W); h = int(rh * H)
    else:
        x, y, w, h = roi_pix
    return clamp_roi(x, y, w, h, W, H)

def load_imgs(folder):
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"): 
        paths += glob.glob(os.path.join(folder, ext))
    imgs = []
    for p in sorted(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: continue
        imgs.append(preprocess(im))
    return (np.stack(imgs,0) if imgs else None)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="Cartella immagini positive (filo presente)")
    ap.add_argument("--neg", required=True, help="Cartella immagini negative (filo assente/errore)")
    ap.add_argument("--save-dir", default="/home/aless/foto", help="Cartella dove salvare gli scatti")
    ap.add_argument("--thr", type=float, default=0.10, help="Soglia LLR per dire 'ok' (>= thr)")
    ap.add_argument("--roi-mode", choices=["relative","pixels"], default="relative")
    ap.add_argument("--roi-rel", nargs=4, type=float, default=[0.47, 0.75, 0.06, 0.08], metavar=("rx","ry","rw","rh"))
    ap.add_argument("--roi-pix", nargs=4, type=int, default=[560, 760, 120, 120], metavar=("x","y","w","h"))
    ap.add_argument("--annotate", action="store_true", help="Salva crop con etichetta OK/ERROR")
    ap.add_argument("--roi-mask", default=None, help="PNG/JPG 8-bit mask da applicare DENTRO il ritaglio")
    args = ap.parse_args()

    SAVE_DIR = args.save_dir
    CROP_DIR = os.path.join(SAVE_DIR, "crops")
    ensure_dir(SAVE_DIR); ensure_dir(CROP_DIR)

    # Carica training set e crea prototipi SULLA ROI (coerenti con il run-time)
    pos_stack_full = load_imgs(args.pos)
    neg_stack_full = load_imgs(args.neg)
    if pos_stack_full is None or neg_stack_full is None:
        raise RuntimeError("Metti esempi nelle cartelle --pos e --neg")

    # Stima dimensione di lavoro dai sample (assumiamo dimensioni simili ai tuoi scatti)
    Ht, Wt = pos_stack_full[0].shape[:2]
    x_t, y_t, w_t, h_t = roi_from_mode(Wt, Ht, args.roi_mode, args.roi_rel, args.roi_pix)
    pos_crops = pos_stack_full[:, y_t:y_t+h_t, x_t:x_t+w_t]
    neg_crops = neg_stack_full[:, y_t:y_t+h_t, x_t:x_t+w_t]

    proto_pos = np.median(pos_crops, axis=0).astype(np.uint8)
    proto_neg = np.median(neg_crops, axis=0).astype(np.uint8)

    # ROI mask opzionale (stessa size del crop a run-time; la ridimensioniamo se serve)
    roi_mask = None
    if args.roi_mask:
        roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            print(f"[WARN] Impossibile leggere la ROI mask: {args.roi_mask}. Procedo senza.")
            roi_mask = None

    # Inizializza camera + bottone
    button = Button(17, pull_up=True, bounce_time=0.05)  # GPIO17 (pin 11)
    picam2 = Picamera2()
    CAM_WIDTH, CAM_HEIGHT = 1280, 960
    still_cfg = picam2.create_still_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "XRGB8888"})
    picam2.configure(still_cfg)
    picam2.start()
    time.sleep(0.3)

    def classify_and_print(crop_bgr, save_stub):
        g = preprocess(crop_bgr)
        hh, ww = g.shape[:2]
        pp = cv2.resize(proto_pos, (ww,hh), interpolation=cv2.INTER_AREA)
        pn = cv2.resize(proto_neg, (ww,hh), interpolation=cv2.INTER_AREA)
        mask_rs = None
        if roi_mask is not None:
            mask_rs = cv2.resize(roi_mask, (ww,hh), interpolation=cv2.INTER_NEAREST)

        s_pos = abs(ncc(g, pp, mask=mask_rs))
        s_neg = ncc(g, pn, mask=mask_rs)
        llr = s_pos - s_neg
        present = (llr >= args.thr)  # True => OK (filo presente)
        print(f"{'ok' if present else 'error'}  LLR={llr:.3f}  (pos={s_pos:.3f} neg={s_neg:.3f})")

        if args.annotate:
            vis = crop_bgr.copy()
            color = (0, 255, 0) if present else (0, 0, 255)
            cv2.putText(
                vis,
                f"{'OK' if present else 'ERROR'}  LLR={llr:.3f}",
                (5, max(18, hh - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(f"{save_stub}_crop_annot.png", vis)

    def scatta_foto():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(SAVE_DIR, f"shot_{ts}.jpg")
        try:
            print(f"[INFO] Scatto -> {filepath}")
            picam2.capture_file(filepath)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[ERR] Scatto non leggibile: {filepath}")
                return
            H, W = img.shape[:2]
            x, y, w, h = roi_from_mode(W, H, args.roi_mode, args.roi_rel, args.roi_pix)
            crop = img[y:y+h, x:x+w]
            crop_path = os.path.join(CROP_DIR, os.path.splitext(os.path.basename(filepath))[0] + "_crop.png")
            cv2.imwrite(crop_path, crop)
            # Classificazione sul crop
            save_stub = os.path.splitext(crop_path)[0]
            classify_and_print(crop, save_stub)
        except Exception as e:
            print(f"[ERR] Errore durante lo scatto/classificazione: {e}")


    # Associa callback
    # Associa callback
    button.when_pressed = scatta_foto

    print("[OK] Pronto. Premi il microinterruttore per scattare e classificare (CTRL+C per uscire).")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        print("\n[OK] Uscita.")

if __name__ == "__main__":
    main()
