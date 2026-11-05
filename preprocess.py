"""AI Hub K-Fashion JSON labels -> MoodFit metadata.csv.

The original images are never cropped here. JSON bbox values are normalized to
[x1, y1, x2, y2]. Cropping is performed later by curation.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "id",
    "image_filename",
    "original_img_path",
    "original_json_path",
    "gender",
    "category",
    "category_detail",
    "mood",
    "bbox",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

K_IMAGE_INFO = "\uc774\ubbf8\uc9c0 \uc815\ubcf4"
K_IMAGE_FILE_NAME = "\uc774\ubbf8\uc9c0 \ud30c\uc77c\uba85"
K_DATASET_DETAIL = "\ub370\uc774\ud130\uc14b \uc0c1\uc138\uc124\uba85"
# AI Hub's K-Fashion schema names this block "렉트좌표" (rectangle coordinates).
K_POSE = "\ub809\ud2b8\uc88c\ud45c"
K_LABELING = "\ub77c\ubca8\ub9c1"
K_STYLE = "\uc2a4\ud0c0\uc77c"
K_CATEGORY = "\uce74\ud14c\uace0\ub9ac"
K_GENDER = "\uc131\ubcc4"
K_X = "X\uc88c\ud45c"
K_Y = "Y\uc88c\ud45c"
K_WIDTH = "\uac00\ub85c"
K_HEIGHT = "\uc138\ub85c"


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_value(data: Any, keys: set[str]) -> Any:
    normalized_keys = {key.lower().replace(" ", "_") for key in keys}
    for item in _walk_dicts(data):
        for key, value in item.items():
            clean_key = str(key).lower().replace(" ", "_")
            if clean_key in normalized_keys and value not in (None, "", []):
                return value
    return None


def _nested_dict(data: Any, key: str) -> dict[str, Any]:
    value = _first_value(data, {key})
    return value if isinstance(value, dict) else {}


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("\uc5ec\uc131", "female", "woman", "women")) or text == "f":
        return "\uc5ec\uc131"
    if any(token in text for token in ("\ub0a8\uc131", "male", "man", "men")) or text == "m":
        return "\ub0a8\uc131"
    return "\uacf5\uc6a9"


def normalize_category(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "top": "\uc0c1\uc758",
        "shirt": "\uc0c1\uc758",
        "blouse": "\uc0c1\uc758",
        "bottom": "\ud558\uc758",
        "pants": "\ud558\uc758",
        "skirt": "\ud558\uc758",
        "outer": "\uc544\uc6b0\ud130",
        "jacket": "\uc544\uc6b0\ud130",
        "coat": "\uc544\uc6b0\ud130",
        "dress": "\uc6d0\ud53c\uc2a4",
        "onepiece": "\uc6d0\ud53c\uc2a4",
        "shoes": "\uc2e0\ubc1c",
        "bag": "\uac00\ubc29",
        "accessory": "\uc561\uc138\uc11c\ub9ac",
    }
    return aliases.get(text.lower(), text or "\uae30\ud0c0")


def normalize_mood(value: Any) -> str:
    if isinstance(value, dict):
        value = next(iter(value.values()), None)
    return str(value or "\ubbf8\ubd84\ub958").strip() or "\ubbf8\ubd84\ub958"


def _coerce_number(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def normalize_bbox(value: Any, bbox_format: str | None = None) -> list[int] | None:
    """Normalize xywh/xyxy bbox values to [x1, y1, x2, y2]."""

    if isinstance(value, dict):
        lower = {str(key).lower(): item for key, item in value.items()}
        if all(key.lower() in lower for key in ("x1", "y1", "x2", "y2")):
            return [_coerce_number(lower[key]) for key in ("x1", "y1", "x2", "y2")]

        x = lower.get("x", lower.get("left", lower.get(K_X.lower())))
        y = lower.get("y", lower.get("top", lower.get(K_Y.lower())))
        width = lower.get("width", lower.get("w", lower.get(K_WIDTH.lower())))
        height = lower.get("height", lower.get("h", lower.get(K_HEIGHT.lower())))
        if None not in (x, y, width, height):
            x, y, width, height = map(_coerce_number, (x, y, width, height))
            return [x, y, x + width, y + height]
        return None

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_coerce_number(item) for item in value]
    if str(bbox_format or "").lower() in {"xyxy", "x1y1x2y2"}:
        return numbers
    x, y, width, height = numbers
    return [x, y, x + width, y + height]


def _find_image_path(json_path: Path, image_dir: Path, payload: Any) -> Path | None:
    """Resolve an image using the JSON stem first, then the JSON file name."""

    raw_name = _first_value(
        payload,
        {"file_name", "filename", "fileName", "image_name", "image_file", "image_path", K_IMAGE_FILE_NAME},
    )
    mood_dir = json_path.parent.name
    candidates: list[Path] = []

    # In the AI Hub split used here, 3725.json maps to mood/3725.jpg even
    # though the JSON metadata's original file name is Amaker_029_11.jpg.
    for extension in IMAGE_EXTENSIONS:
        candidates.extend(
            [
                image_dir / mood_dir / f"{json_path.stem}{extension}",
                image_dir / f"{json_path.stem}{extension}",
            ]
        )

    if raw_name:
        raw_path = Path(str(raw_name))
        candidates.extend(
            [
                json_path.parent / raw_path,
                image_dir / mood_dir / raw_path.name,
                image_dir / raw_path,
                image_dir / raw_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(image_dir.rglob(f"{json_path.stem}.*"))
    return next((path.resolve() for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS), None)


def _label_detail(labeling: dict[str, Any], category_key: str) -> str:
    entries = labeling.get(category_key, []) if isinstance(labeling, dict) else []
    if not isinstance(entries, list):
        entries = [entries]
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get(K_CATEGORY) or entry.get("category") or entry.get("class")
            if value:
                return str(value)
    return ""


def _aihub_annotations(payload: Any) -> list[dict[str, Any]]:
    detail = _nested_dict(payload, K_DATASET_DETAIL)
    pose = detail.get(K_POSE, {}) if detail else {}
    labeling = detail.get(K_LABELING, {}) if detail else {}
    annotations: list[dict[str, Any]] = []

    if isinstance(pose, dict):
        for category_key, values in pose.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                bbox = normalize_bbox(value)
                if bbox:
                    annotations.append(
                        {
                            "category": category_key,
                            "category_detail": _label_detail(labeling, category_key),
                            "bbox": bbox,
                        }
                    )
    return annotations


def _fallback_annotations(payload: Any) -> list[dict[str, Any]]:
    result = []
    for item in _walk_dicts(payload):
        bbox = item.get("bbox", item.get("bounding_box", item.get("box")))
        if bbox is not None:
            result.append({"category": item.get("category", item.get("class", "")), "bbox": normalize_bbox(bbox)})
    return [item for item in result if item.get("bbox")]


def _safe_id(value: str, fallback: int) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return cleaned or str(fallback)


def process_aihub_data(
    json_dir: str | Path,
    img_dir: str | Path,
    output_csv_path: str | Path,
) -> pd.DataFrame:
    """Parse AI Hub labels and write a lightweight metadata CSV."""

    json_root = Path(json_dir)
    image_root = Path(img_dir)
    output_path = Path(output_csv_path)
    project_root = Path.cwd().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for json_path in sorted(json_root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        image_path = _find_image_path(json_path, image_root, payload)
        if image_path is None:
            continue

        def portable_path(path: Path) -> str:
            """Store paths relative to the repository for cross-PC use."""
            try:
                return path.resolve().relative_to(project_root).as_posix()
            except ValueError:
                return path.resolve().as_posix()

        image_name = str(_first_value(payload, {K_IMAGE_FILE_NAME, "file_name", "filename"}) or image_path.name)
        image_info = _nested_dict(payload, K_IMAGE_INFO)
        image_id = image_info.get("\uc774\ubbf8\uc9c0 \uc2dd\ubcc4\uc790", json_path.stem)
        mood = json_path.parent.name or normalize_mood(_first_value(payload, {"mood", "style", K_STYLE}))
        gender = normalize_gender(_first_value(payload, {"gender", "sex", K_GENDER}))
        annotations = _aihub_annotations(payload) or _fallback_annotations(payload) or [{"category": "", "bbox": None}]

        for annotation_index, annotation in enumerate(annotations):
            category = normalize_category(annotation.get("category", ""))
            category_detail = normalize_category(annotation.get("category_detail", "")) if annotation.get("category_detail") else category
            source_id = json_path.stem if len(annotations) == 1 else f"{json_path.stem}_{annotation_index}"
            records.append(
                {
                    "id": _safe_id(str(source_id), len(records) + 1),
                    "image_filename": image_name,
                    "original_img_path": portable_path(image_path),
                    "original_json_path": portable_path(json_path),
                    "gender": gender,
                    "category": category,
                    "category_detail": category_detail,
                    "mood": normalize_mood(mood),
                    "bbox": json.dumps(annotation.get("bbox"), ensure_ascii=False) if annotation.get("bbox") else "",
                }
            )

    dataframe = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"metadata generated: {output_path} ({len(dataframe)} rows)")
    return dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MoodFit metadata.csv from AI Hub JSON labels")
    parser.add_argument("--json-dir", default="recommend_pool/raw_data/labels")
    parser.add_argument("--image-dir", default="recommend_pool/raw_data/images")
    parser.add_argument("--output", default="recommend_pool/metadata.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_aihub_data(args.json_dir, args.image_dir, args.output)


if __name__ == "__main__":
    main()
