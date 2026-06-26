"""Konvertiert den IP102-Detection-Datensatz (PASCAL-VOC-Format) in das
YOLO-Trainingsformat (images/ + labels/ mit normalisierten Bounding-Boxen).

Voraussetzung: Du hast den IP102-Datensatz bereits manuell heruntergeladen.
Offizielle Quelle: https://github.com/xpwu95/IP102 (Verteilung über Google
Drive/Baidu Pan, ein automatischer Download ist nicht möglich).

Die Drive-Kopie von IP102/Detection/VOC2007 liefert `Annotations.tar` und
`JPEGImages.tar` statt fertiger Ordner - diese zuerst entpacken:

    cd IP102/Detection/VOC2007
    tar -xf Annotations.tar
    tar -xf JPEGImages.tar

Erwartete Eingabestruktur danach:

    <input-dir>/
        Annotations/*.xml
        JPEGImages/*.jpg
        ImageSets/Main/trainval.txt, test.txt   (Bild-IDs ohne Dateiendung, eine pro Zeile)

Eine `classes.txt` wird NICHT vorausgesetzt: falls keine vorhanden ist,
werden die Klassennamen automatisch aus den `<name>`-Tags aller
Annotationen ermittelt (alphabetisch sortiert für stabile IDs).

Beispiel:
    python scripts/prepare_ip102.py \
        --input-dir /pfad/zu/IP102/Detection/VOC2007 \
        --output-dir data/ip102_yolo
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def load_split_ids(split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(xml_path: Path) -> tuple[float, float, list[tuple[str, float, float, float, float]]]:
    """Liest eine VOC-XML-Annotation. Gibt (Breite, Höhe, [(Name, xmin, ymin, xmax, ymax), ...]) zurück."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return img_w, img_h, objects


def discover_class_names(input_dir: Path, image_ids: list[str]) -> list[str]:
    classes_file = input_dir / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    names: set[str] = set()
    for image_id in image_ids:
        xml_path = input_dir / "Annotations" / f"{image_id}.xml"
        if not xml_path.exists():
            continue
        _, _, objects = parse_annotation(xml_path)
        names.update(name for name, *_ in objects)
    return sorted(names)


def to_yolo_lines(img_w: float, img_h: float, objects: list[tuple[str, float, float, float, float]], class_to_id: dict[str, int]) -> list[str]:
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        if name not in class_to_id:
            continue  # unbekannte Klasse -> überspringen
        class_id = class_to_id[name]
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

        img_w, img_h, objects = parse_annotation(xml_path)
        lines = to_yolo_lines(img_w, img_h, objects, class_to_id)
        (labels_out / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        shutil.copy(image_path, images_out / f"{image_id}.jpg")

    print(f"[{split_name}] {len(image_ids) - skipped} Bilder konvertiert, {skipped} übersprungen (fehlende Dateien)")


def main() -> None:
    parser = argparse.ArgumentParser(description="IP102 (VOC-Format) -> YOLO-Trainingsformat")
    parser.add_argument("--input-dir", required=True, type=Path, help="Pfad zum entpackten IP102 Detection/VOC2007 Ordner")
    parser.add_argument("--output-dir", default=Path("data/ip102_yolo"), type=Path)
    parser.add_argument(
        "--train-split-file", default="trainval.txt",
        help="Datei in ImageSets/Main, die als YOLO-'train'-Split verwendet wird",
    )
    parser.add_argument(
        "--val-split-file", default="test.txt",
        help="Datei in ImageSets/Main, die als YOLO-'val'-Split verwendet wird",
    )
    args = parser.parse_args()

    splits = {"train": args.train_split_file, "val": args.val_split_file}
    split_ids: dict[str, list[str]] = {}
    for yolo_name, filename in splits.items():
        split_file = args.input_dir / "ImageSets" / "Main" / filename
        if not split_file.exists():
            print(f"Überspringe Split '{yolo_name}': {split_file} nicht gefunden")
            continue
        split_ids[yolo_name] = load_split_ids(split_file)

    all_ids = [image_id for ids in split_ids.values() for image_id in ids]
    class_names = discover_class_names(args.input_dir, all_ids)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"{len(class_names)} Klassen gefunden")

    for yolo_name, image_ids in split_ids.items():
        convert_split(yolo_name, image_ids, args.input_dir, args.output_dir, class_to_id)

    yaml_path = args.output_dir / "ip102.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text(
        f"path: {args.output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"\nYOLO-Datenkonfiguration geschrieben: {yaml_path}")


if __name__ == "__main__":
    main()
