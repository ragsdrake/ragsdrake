"""Konvertiert einen selbst ausgewählten Teil des Lepidoptera-Datensatzes
(Schmetterlinge & Motten, figshare, DOI 10.25452/figshare.plus.29135618) in
das YOLO-Trainingsformat (images/ + labels/ mit normalisierten Bounding-Boxen).

Der Datensatz liegt dort als Klassifikations-Datensatz vor: ein ZIP pro Art,
jedes Bild hat nur ein Art-Label fürs ganze Bild - keine Bounding-Box-
Annotationen. Da unsere Pipeline ein YOLO-Detektionsmodell nutzt, behandelt
dieses Skript jedes Bild als eine einzige, fast bildfüllende Bounding-Box.
Das ist eine Näherung (kein Ersatz für echte Box-Annotationen wie bei
IP102), aber für die meist recht zentrierten Naturfotos der Citizen-
Science-App ("Schmetterlinge Österreichs") brauchbar.

Voraussetzung: Du hast dir manuell ein paar Arten-ZIPs von der figshare-Seite
heruntergeladen und in einem gemeinsamen Ordner entpackt, einen Unterordner
pro Art (Ordnername = Artname):

    <input-dir>/
        Aglais io/*.jpg
        Vanessa cardui/*.jpg
        ...

Beispiel:
    python3 scripts/prepare_lepidoptera.py \
        --input-dir ~/Downloads/lepidoptera_arten \
        --output-dir data/lepidoptera_yolo
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Ganzes Bild als Box, leicht eingerückt statt exakt auf den Rand (0/1).
FULL_FRAME_BOX = "0.500000 0.500000 0.980000 0.980000"

# Unter dieser Bildanzahl pro Art wandert alles in den Trainings-Split -
# bei zu wenigen Bildern ist ein eigener Val-Anteil nicht sinnvoll.
MIN_IMAGES_FOR_VAL_SPLIT = 5


def discover_species(input_dir: Path) -> list[str]:
    return sorted(p.name for p in input_dir.iterdir() if p.is_dir())


def collect_images(species_dir: Path) -> list[Path]:
    return sorted(p for p in species_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def add_background_images(bg_dir: Path, output_dir: Path, val_ratio: float, rng: random.Random) -> int:
    """Kopiert Nicht-Schmetterling-Bilder mit leerer Labeldatei in den Datensatz.

    YOLO erkennt leere .txt-Dateien als Hintergrundbilder (kein Objekt vorhanden)
    und lernt dadurch, bei fremden Motiven keine Erkennung auszugeben.
    """
    images = sorted(p for p in bg_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"Warnung: keine Bilder in --background-dir {bg_dir} gefunden")
        return 0

    rng.shuffle(images)
    n_val = max(1, round(len(images) * val_ratio)) if len(images) >= MIN_IMAGES_FOR_VAL_SPLIT else 0
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for split_name, split_images in splits.items():
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            stem = f"bg_{image_path.stem}"
            shutil.copy(image_path, images_out / f"{stem}{image_path.suffix.lower()}")
            (labels_out / f"{stem}.txt").write_text("", encoding="utf-8")  # leer = kein Objekt

    total = len(images)
    print(f"  Hintergrundbilder: {total - n_val} train, {n_val} val (leere Labels)")
    return total


def convert(input_dir: Path, output_dir: Path, val_ratio: float, seed: int, background_dir: Path | None = None) -> None:
    species_names = discover_species(input_dir)
    if not species_names:
        raise SystemExit(f"Keine Art-Unterordner in {input_dir} gefunden")

    class_to_id = {name: idx for idx, name in enumerate(species_names)}
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}

    for species in species_names:
        images = collect_images(input_dir / species)
        if not images:
            print(f"Warnung: keine Bilder für '{species}' gefunden, übersprungen")
            continue

        rng.shuffle(images)
        n_val = max(1, round(len(images) * val_ratio)) if len(images) >= MIN_IMAGES_FOR_VAL_SPLIT else 0
        splits = {"val": images[:n_val], "train": images[n_val:]}

        class_id = class_to_id[species]
        for split_name, split_images in splits.items():
            images_out = output_dir / "images" / split_name
            labels_out = output_dir / "labels" / split_name
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                stem = f"{species.replace(' ', '_')}_{image_path.stem}"
                shutil.copy(image_path, images_out / f"{stem}{image_path.suffix.lower()}")
                (labels_out / f"{stem}.txt").write_text(f"{class_id} {FULL_FRAME_BOX}\n", encoding="utf-8")
                counts[split_name] += 1

        print(f"  {species}: {len(splits['train'])} train, {len(splits['val'])} val")

    if background_dir is not None:
        add_background_images(background_dir, output_dir, val_ratio, rng)

    print(f"\n{len(species_names)} Arten, {counts['train']} Trainings- / {counts['val']} Val-Bilder konvertiert")

    yaml_path = output_dir / "lepidoptera.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(species_names))
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"YOLO-Datenkonfiguration geschrieben: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lepidoptera-Artendatensatz (figshare) -> YOLO-Trainingsformat")
    parser.add_argument("--input-dir", required=True, type=Path, help="Ordner mit einem Unterordner pro Art")
    parser.add_argument("--output-dir", default=Path("data/lepidoptera_yolo"), type=Path)
    parser.add_argument("--val-ratio", default=0.1, type=float, help="Anteil je Art, der als Val-Split genutzt wird")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--background-dir", default=None, type=Path,
        help="Ordner mit beliebigen Nicht-Schmetterling-Bildern (Fußball, Himmel, ...). "
             "Diese werden mit leerer Labeldatei eingefügt, damit das Modell bei fremden "
             "Motiven keine Erkennung ausgibt.",
    )
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir, args.val_ratio, args.seed, args.background_dir)


if __name__ == "__main__":
    main()
