#!/usr/bin/env bash
# Trainiert ein YOLOv8-Modell auf dem konvertierten Lepidoptera-Datensatz.
# Voraussetzung: scripts/prepare_lepidoptera.py wurde bereits ausgeführt.
#
# Nutzung:
#   scripts/train_lepidoptera.sh [data.yaml] [epochs] [imgsz]
set -euo pipefail

DATA_YAML="${1:-data/lepidoptera_yolo/lepidoptera.yaml}"
EPOCHS="${2:-100}"
IMGSZ="${3:-640}"

yolo train \
  data="$DATA_YAML" \
  model=yolov8n.pt \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  project=runs/lepidoptera \
  name=train

echo
echo "Fertig trainiertes Modell liegt unter: runs/lepidoptera/train/weights/best.pt"
echo "Zum Einbinden: Pfad in configs/default.yaml unter detector.weights eintragen."
