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
Annotationen ermittelt. IP102 speichert dort typischerweise nur die
numerische Pest-ID (1-102) statt eines Artnamens - diese IDs werden dann
über eine eingebettete Tabelle (offizielle IP102-classes.txt) in echte
Artnamen übersetzt, numerisch sortiert für stabile IDs.

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


class AnnotationError(Exception):
    pass


# IP102s VOC-Annotationen speichern im <name>-Tag nur die numerische Pest-ID
# (1-102) statt eines Artnamens - die echten Namen stehen in der offiziellen
# classes.txt der Classification-Variante (github.com/xpwu95/IP102), die der
# Detection/VOC2007-Ordner i. d. R. nicht enthält. Diese Tabelle übersetzt die
# IDs zurück in lesbare Artnamen.
IP102_CLASS_NAMES: dict[int, str] = {
    1: "rice leaf roller", 2: "rice leaf caterpillar", 3: "paddy stem maggot",
    4: "asiatic rice borer", 5: "yellow rice borer", 6: "rice gall midge",
    7: "Rice Stemfly", 8: "brown plant hopper", 9: "white backed plant hopper",
    10: "small brown plant hopper", 11: "rice water weevil", 12: "rice leafhopper",
    13: "grain spreader thrips", 14: "rice shell pest", 15: "grub",
    16: "mole cricket", 17: "wireworm", 18: "white margined moth",
    19: "black cutworm", 20: "large cutworm", 21: "yellow cutworm",
    22: "red spider", 23: "corn borer", 24: "army worm", 25: "aphids",
    26: "Potosiabre vitarsis", 27: "peach borer", 28: "english grain aphid",
    29: "green bug", 30: "bird cherry-oataphid", 31: "wheat blossom midge",
    32: "penthaleus major", 33: "longlegged spider mite", 34: "wheat phloeothrips",
    35: "wheat sawfly", 36: "cerodonta denticornis", 37: "beet fly",
    38: "flea beetle", 39: "cabbage army worm", 40: "beet army worm",
    41: "Beet spot flies", 42: "meadow moth", 43: "beet weevil",
    44: "sericaorient alismots chulsky", 45: "alfalfa weevil", 46: "flax budworm",
    47: "alfalfa plant bug", 48: "tarnished plant bug", 49: "Locustoidea",
    50: "lytta polita", 51: "legume blister beetle", 52: "blister beetle",
    53: "therioaphis maculata Buckton", 54: "odontothrips loti", 55: "Thrips",
    56: "alfalfa seed chalcid", 57: "Pieris canidia", 58: "Apolygus lucorum",
    59: "Limacodidae", 60: "Viteus vitifoliae", 61: "Colomerus vitis",
    62: "Brevipoalpus lewisi McGregor", 63: "oides decempunctata",
    64: "Polyphagotars onemus latus", 65: "Pseudococcus comstocki Kuwana",
    66: "parathrene regalis", 67: "Ampelophaga", 68: "Lycorma delicatula",
    69: "Xylotrechus", 70: "Cicadella viridis", 71: "Miridae",
    72: "Trialeurodes vaporariorum", 73: "Erythroneura apicalis",
    74: "Papilio xuthus", 75: "Panonchus citri McGregor",
    76: "Phyllocoptes oleiverus ashmead", 77: "Icerya purchasi Maskell",
    78: "Unaspis yanonensis", 79: "Ceroplastes rubens", 80: "Chrysomphalus aonidum",
    81: "Parlatoria zizyphus Lucus", 82: "Nipaecoccus vastalor",
    83: "Aleurocanthus spiniferus", 84: "Tetradacus c Bactrocera minax",
    85: "Dacus dorsalis(Hendel)", 86: "Bactrocera tsuneonis", 87: "Prodenia litura",
    88: "Adristyrannus", 89: "Phyllocnistis citrella Stainton",
    90: "Toxoptera citricidus", 91: "Toxoptera aurantii",
    92: "Aphis citricola Vander Goot", 93: "Scirtothrips dorsalis Hood",
    94: "Dasineura sp", 95: "Lawana imitata Melichar", 96: "Salurnis marginella Guerr",
    97: "Deporaus marginatus Pascoe", 98: "Chlumetia transversa",
    99: "Mango flat beak leafhopper", 100: "Rhytidodera bowrinii white",
    101: "Sternochetus frigidus", 102: "Cicadellidae",
}


def parse_annotation(xml_path: Path) -> tuple[float, float, list[tuple[str, float, float, float, float]]]:
    """Liest eine VOC-XML-Annotation. Gibt (Breite, Höhe, [(Name, xmin, ymin, xmax, ymax), ...]) zurück.

    Manche IP102-Annotationsdateien enthalten nach dem schließenden
    </annotation>-Tag noch zusätzlichen (oft doppelten) Inhalt, der die
    Datei als XML ungültig macht ("junk after document element"). Wir
    schneiden deshalb alles nach dem ersten </annotation> ab, bevor wir
    parsen, statt die Datei komplett zu verwerfen.
    """
    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    end_idx = text.find("</annotation>")
    if end_idx != -1:
        text = text[: end_idx + len("</annotation>")]

    try:
        root = ET.fromstring(text)
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
    except (ET.ParseError, AttributeError, TypeError, ValueError) as exc:
        raise AnnotationError(f"{xml_path.name}: {exc}") from exc


def discover_class_names(input_dir: Path, image_ids: list[str]) -> list[tuple[str, str]]:
    """Ermittelt die in den Annotationen vorkommenden Klassen.

    Gibt (Roh-Label-aus-XML, Anzeigename) zurück, in stabiler Reihenfolge.
    Roh-Label ist exakt der Text aus dem <name>-Tag (wird für die spätere
    Zuordnung der Bounding-Boxen benötigt); Anzeigename ist - falls möglich -
    der echte Artname statt der nackten IP102-Nummer.
    """
    classes_file = input_dir / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [(name, name) for name in names]

    raw_labels: set[str] = set()
    for image_id in image_ids:
        xml_path = input_dir / "Annotations" / f"{image_id}.xml"
        if not xml_path.exists():
            continue
        try:
            _, _, objects = parse_annotation(xml_path)
        except AnnotationError as exc:
            print(f"Warnung: defekte Annotation übersprungen ({exc})")
            continue
        raw_labels.update(name for name, *_ in objects)

    if raw_labels and all(label.isdigit() for label in raw_labels):
        # IP102-typisch: <name> enthält die numerische Pest-ID statt eines
        # Artnamens -> über IP102_CLASS_NAMES auf echte Namen abbilden und
        # numerisch (nicht alphabetisch als String) sortieren.
        sorted_labels = sorted(raw_labels, key=int)
        return [(label, IP102_CLASS_NAMES.get(int(label), label)) for label in sorted_labels]
    return [(label, label) for label in sorted(raw_labels)]


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

        try:
            img_w, img_h, objects = parse_annotation(xml_path)
        except AnnotationError as exc:
            print(f"Warnung: defekte Annotation übersprungen ({exc})")
            skipped += 1
            continue

        lines = to_yolo_lines(img_w, img_h, objects, class_to_id)
        (labels_out / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        shutil.copy(image_path, images_out / f"{image_id}.jpg")

    print(f"[{split_name}] {len(image_ids) - skipped} Bilder konvertiert, {skipped} übersprungen (fehlend/defekt)")


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
    class_pairs = discover_class_names(args.input_dir, all_ids)
    class_to_id = {raw_label: idx for idx, (raw_label, _display_name) in enumerate(class_pairs)}
    class_names = [display_name for _raw_label, display_name in class_pairs]
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
