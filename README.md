# Insekten-Tracker

Kamerabasierte Erkennung und Tracking verschiedener Insektenarten.
Modular aufgebaut, damit dieselbe Pipeline auf drei Zielplattformen läuft:

- **Raspberry Pi / Embedded** (Pi-Kameramodul, dauerhafter Betrieb im Feld)
- **PC/Laptop** (Webcam, lokale Entwicklung/Tests)
- **Server** (Video-Uploads über das Dashboard, nachträgliche Analyse)

## Architektur

```
Frame-Quelle  ->  Detektor  ->  Tracker  ->  Speicherung (SQLite)  ->  Dashboard (FastAPI)
(capture/)        (detection/)  (tracking/)  (storage/)                (api/)
```

- `capture/` – `FrameSource`-Schnittstelle mit Implementierungen für Webcam,
  Pi-Kamera und Videodatei. Austauschbar über `configs/default.yaml` (`source.type`).
- `detection/` – `Detector`-Schnittstelle, aktuell ein YOLO-Wrapper (Ultralytics).
- `tracking/` – einfacher, abhängigkeitsfreier IOU-Tracker, der Erkennungen über
  Frames hinweg zu Tracks mit stabiler ID zusammenführt (austauschbar gegen
  ByteTrack/DeepSORT, ohne die Pipeline anzufassen).
- `storage/` – SQLite-Persistenz der Sichtungen (`sightings`-Tabelle).
- `pipeline.py` – verbindet die Schichten zu einem Lauf.
- `api/` – FastAPI-Dashboard (Live-Artenverteilung, letzte Sichtungen, Video-Upload).

### Wichtiger Hinweis zum Erkennungsmodell

Die Standardgewichte (`yolov8n.pt`) sind ein allgemeines COCO-Modell und
**erkennen keine Insektenarten** – sie sind nur ein lauffähiger Platzhalter
für das Grundgerüst. Für echte Artenerkennung muss ein YOLO-Modell auf einem
Insekten-Datensatz feinabgestimmt werden (z. B. [IP102](https://github.com/xpwu95/IP102)
oder Insekten-Datensätze auf [Roboflow](https://roboflow.com/) /
[iNaturalist](https://www.inaturalist.org/)) und über `detector.weights` in
`configs/default.yaml` eingebunden werden.

## Eigenes Modell auf IP102 trainieren

IP102 ist nur ein Bild-Datensatz, kein fertiges Modell. Um ihn zu nutzen,
musst du selbst ein YOLO-Modell darauf trainieren; danach lässt sich das
Ergebnis (`best.pt`) wie jedes andere YOLO-Modell einbinden.

1. **Datensatz herunterladen** (manuell, kein automatischer Download
   möglich): offizielle Quelle [github.com/xpwu95/IP102](https://github.com/xpwu95/IP102),
   verteilt über Google Drive/Baidu Pan. Du brauchst die **Detection**-Variante
   (PASCAL-VOC-Format mit Bounding-Boxen), nicht nur die Klassifikations-Variante.

2. **In YOLO-Format konvertieren**:

   ```bash
   python scripts/prepare_ip102.py \
     --input-dir /pfad/zu/IP102/Detection/VOC2007 \
     --output-dir data/ip102_yolo
   ```

   Das Skript erwartet die Standard-VOC-Struktur (`Annotations/`, `JPEGImages/`,
   `ImageSets/Main/{train,val,test}.txt`, `classes.txt`). Falls deine Kopie
   des Datensatzes anders benannt ist, Ordner entsprechend anpassen.

3. **Trainieren**:

   ```bash
   scripts/train_ip102.sh data/ip102_yolo/ip102.yaml 100 640
   ```

   Läuft über die Ultralytics-CLI (`yolo train ...`), Dauer abhängig von
   Hardware (Stunden bis Tage auf CPU, deutlich schneller mit GPU). Ergebnis
   liegt danach unter `runs/ip102/train/weights/best.pt`.

4. **Einbinden**: Pfad zur trainierten `best.pt` in `configs/default.yaml`
   unter `detector.weights` eintragen. Ab dann nutzt die Pipeline (Webcam,
   Pi-Kamera, Video-Upload) automatisch das trainierte Modell.

Das trainierte Modell erkennt nur die ~102 IP102-Schädlingsarten zuverlässig;
Arten außerhalb dieses Datensatzes werden nicht oder falsch erkannt.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### PC/Laptop mit Webcam

```bash
insect-tracker --source webcam
```

### Raspberry Pi mit Kameramodul

```bash
sudo apt install -y python3-picamera2
pip install -e ".[pi]"
insect-tracker --source picamera
```

### Server (Video-Upload über Dashboard)

```bash
insect-tracker-api
# Dashboard unter http://localhost:8000 öffnen und Video hochladen
```

Videodateien lassen sich auch direkt per CLI verarbeiten:

```bash
insect-tracker --source video_file --video-path pfad/zum/video.mp4
```

## Dashboard

`insect-tracker-api` startet das Web-Dashboard (Artenverteilung als Chart,
Tabelle der letzten Sichtungen, Video-Upload-Formular) unter
`http://localhost:8000`.

## Tests

```bash
pytest
```

## Roadmap

- [ ] Eigenes Insekten-Erkennungsmodell auf IP102 trainieren (Skripte vorhanden, Training noch ausstehend)
- [ ] Hintergrund-Queue für Video-Uploads (statt synchroner Verarbeitung)
- [ ] Robusterer Tracker (ByteTrack/DeepSORT) als Alternative zum IOU-Tracker
- [ ] Live-Video-Stream im Dashboard (nicht nur Statistiken)
- [ ] Export von Sichtungsdaten (CSV/JSON)
