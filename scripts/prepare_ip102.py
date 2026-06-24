"""Konvertiert den IP102-Detection-Datensatz (PASCAL-VOC-Format) in das
YOLO-Trainingsformat (images/ + labels/ mit normalisierten Bounding-Boxen).

Voraussetzung: Du hast den IP102-Datensatz bereits manuell heruntergeladen
und entpackt. Offizielle Quelle: https://github.com/xpwu95/IP102
(Verteilung über Google Drive/Baidu Pan, ein automatischer Download ist
nicht möglich).

Erwartete Eingabestruktur (PASCAL-VOC, wie im IP102-Detection-Release):

    <input-dir>/
        Annotations/*.xml
        JPEGImages/*.jpg
        ImageSets/Main/{train,val,test}.txt   (Bild-IDs ohne Dateiendung, eine pro Zeile)
        classes.txt                            (eine Klasse pro Zeile, Zeilennummer = Klassen-ID)

Falls deine Kopie des Datensatzes anders benannt/strukturiert ist, die
Ordner entsprechend umbenennen oder dieses Skript anpassen.

Beispiel:
    python scripts/prepare_ip102.py \
        --input-dir /pfad/zu/IP102/Detection/VOC2007 \
        --output-dir data/ip102_yolo
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def load_class_names(classes_file: Path) -> list[str]:
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split_ids(split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_annotation(xml_path: Path, class_to_id: dict[str, int]) -> list[str]:
    """Liest eine VOC-XML-Annotation und gibt YOLO-Zeilen zurück
    (class_id x_center y_center width height, normalisiert auf [0, 1])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_to_id:
            continue  # Klasse nicht in classes.txt -> überspringen
        class_id = class_to_id[name]

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def convert_split(
    split_name: str,
    image_ids: list[str],
    input_dir: Path,
    output_dir: Path,
    class_to_id: dict[str, int],
) -> None:
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for image_id in image_ids:
        xml_path = input_dir / "Annotations" / f"{image_id}.xml"
        image_path = input_dir / "JPEGImages" / f"{image_id}.jpg"
        if not xml_path.exists() or not image_path.exists():
            skipped += 1
            continue

        lines = convert_annotation(xml_path, class_to_id)
        (labels_out / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        shutil.copy(image_path, images_out / f"{image_id}.jpg")

    print(f"[{split_name}] {len(image_ids) - skipped} Bilder konvertiert, {skipped} übersprungen (fehlende Dateien)")


def main() -> None:
    parser = argparse.ArgumentParser(description="IP102 (VOC-Format) -> YOLO-Trainingsformat")
    parser.add_argument("--input-dir", required=True, type=Path, help="Pfad zum entpackten IP102 Detection/VOC2007 Ordner")
    parser.add_argument("--output-dir", default=Path("data/ip102_yolo"), type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    class_names = load_class_names(args.input_dir / "classes.txt")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for split in args.splits:
        split_file = args.input_dir / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            print(f"Überspringe Split '{split}': {split_file} nicht gefunden")
            continue
        image_ids = load_split_ids(split_file)
        convert_split(split, image_ids, args.input_dir, args.output_dir, class_to_id)

    yaml_path = args.output_dir / "ip102.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text(
        f"path: {args.output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"\nYOLO-Datenkonfiguration geschrieben: {yaml_path}")


if __name__ == "__main__":
    main()
