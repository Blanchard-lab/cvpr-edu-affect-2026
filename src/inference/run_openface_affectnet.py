import argparse
import csv
import sys
from pathlib import Path

import cv2
print(cv2.__version__)
print(cv2.data.haarcascades)
import numpy as np
import torch


def load_image_rgb(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(path, img)
    return img

def to_tensor(face): 
    return torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def softmax_np(x: np.ndarray):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--openface_root", required=True)
    ap.add_argument("--weights_path", required=True)
    args = ap.parse_args()

    sys.path.append(args.openface_root)
    from model.MLT import MLT

    device = args.device
    model = MLT()
    state = torch.load(args.weights_path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["pred_label", "pred_idx"]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        emotion_map = {'Neutral':0, 'Happy':0, 'Sad':0, 'Surprise':0, 'Fear':0, 'Disgust':0, 'Anger':0, 'Contempt':0}
        for row in rows:
            img = load_image_rgb(row["image_path"])
            if img is None:
                continue
            print("SHAPE:",img.shape)    
            x = to_tensor(img).to(device)

            with torch.no_grad():
                emotion_logits, _, _, _ = model(x)

            logits_np = emotion_logits.detach().cpu().numpy()[0]
            probs = softmax_np(logits_np)
            pred_idx = int(np.argmax(probs))
            print(pred_idx)
            pred_label = emotion_classes[pred_idx]

            out_row = dict(row)
            out_row["pred_label"] = pred_label
            out_row["pred_idx"] = pred_idx
            emotion_map[pred_label] += 1
            writer.writerow(out_row)
        print(emotion_map)
    print(f"Wrote: {out_csv}")
    

if __name__ == "__main__":
    main()