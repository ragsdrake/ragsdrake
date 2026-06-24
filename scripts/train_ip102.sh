#!/usr/bin/env bash
# Trainiert ein YOLOv8-Modell auf dem konvertierten IP102-Datensatz.
# Voraussetzung: scripts/prepare_ip102.py wurde bereits ausgeführt.
#
# Nutzung:
#   scripts/train_ip102.sh [data.yaml] [epochs] [imgsz]
set -euo pipefail

DATA_YAML="${1:-data/ip102_yolo/ip102.yaml}"
EPOCHS="${2:-100}"
IMGSZ="${3:-640}"

yolo train \
  data="$DATA_YAML" \
  model=yolov8n.pt \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  project=runs/ip102 \
  name=train

echo
echo "Fertig trainiertes Modell liegt unter: runs/ip102/train/weights/best.pt"
echo "Zum Einbinden: Pfad in configs/default.yaml unter detector.weights eintragen."
