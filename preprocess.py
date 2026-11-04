"""AI Hub K-Fashion JSON 라벨을 metadata.csv로 정규화한다.

이 모듈은 원본 이미지를 절대 미리 크롭하지 않는다. bbox는 CSV에 JSON 문자열로
기록하고, 실제 크롭은 curation.lazy_crop_item이 화면 렌더링 시 수행한다.
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
    "original_img_path",
    "original_json_path",
    "gender",
    "category",
    "mood",
    "bbox",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("여성", "female", "woman", "women")) or text == "f":
        return "여성"
    if any(token in text for token in ("남성", "male", "man", "men")) or text == "m":
        return "남성"
    return "공용"


def normalize_category(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "top": "상의",
        "shirt": "상의",
        "blouse": "상의",
        "bottom": "하의",
        "pants": "하의",
        "skirt": "하의",
        "outer": "아우터",
        "jacket": "아우터",
        "coat": "아우터",
        "dress": "원피스",
        "onepiece": "원피스",
        "shoes": "신발",
        "bag": "가방",
        "accessory": "액세서리",
    }
    return aliases.get(text.lower(), text or "기타")


def normalize_mood(value: Any) -> str:
    if isinstance(value, dict):
        value = next(iter(value.values()), None)
    return str(value or "미분류").strip() or "미분류"


def _coerce_number(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def normalize_bbox(value: Any, bbox_format: str | None = None) -> list[int] | None:
    """bbox를 [x1, y1, x2, y2]로 변환한다."""

    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        if all(key in lower for key in ("x1", "y1", "x2", "y2")):
            return [_coerce_number(lower[key]) for key in ("x1", "y1", "x2", "y2")]
        x = lower.get("x", lower.get("left"))
        y = lower.get("y", lower.get("top"))
        width = lower.get("width", lower.get("w"))
        height = lower.get("height", lower.get("h"))
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
    raw_name = _first_value(
        payload,
        {"file_name", "filename", "fileName", "image_name", "image_file", "image_path"},
    )
    candidates: list[Path] = []
    if raw_name:
        raw_path = Path(str(raw_name))
        candidates.extend([json_path.parent / raw_path, image_dir / raw_path, image_dir / raw_path.name])
    candidates.extend(image_dir / f"{json_path.stem}{extension}" for extension in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(image_dir.rglob(f"{json_path.stem}.*"))
    return next((path.resolve() for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS), None)


def _annotation_dicts(payload: Any) -> list[dict[str, Any]]:
    result = []
    for item in _walk_dicts(payload):
        bbox = item.get("bbox", item.get("bounding_box", item.get("box")))
        if bbox is not None:
            result.append(item)
    return result


def _safe_id(value: str, fallback: int) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return cleaned or str(fallback)


def process_aihub_data(
    json_dir: str | Path,
    img_dir: str | Path,
    output_csv_path: str | Path,
) -> pd.DataFrame:
    """JSON 라벨을 순회해 metadata CSV를 생성하고 DataFrame을 반환한다."""

    json_root = Path(json_dir)
    image_root = Path(img_dir)
    output_path = Path(output_csv_path)
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
        annotations = _annotation_dicts(payload) or [{}]
        root_gender = _first_value(payload, {"gender", "sex", "성별"})
        root_category = _first_value(payload, {"category", "item", "clothes", "품목"})
        root_mood = _first_value(payload, {"mood", "style", "leisure_wear", "무드"})

        for annotation_index, annotation in enumerate(annotations):
            bbox_value = annotation.get("bbox", annotation.get("bounding_box", annotation.get("box")))
            bbox = normalize_bbox(bbox_value, annotation.get("bbox_format"))
            category = annotation.get("category", annotation.get("class", root_category))
            gender = annotation.get("gender", annotation.get("sex", root_gender))
            mood = annotation.get("mood", annotation.get("style", root_mood))
            source_id = f"{json_path.stem}_{annotation_index}" if len(annotations) > 1 else json_path.stem
            records.append(
                {
                    "id": _safe_id(source_id, len(records) + 1),
                    "original_img_path": str(image_path),
                    "original_json_path": str(json_path.resolve()),
                    "gender": normalize_gender(gender),
                    "category": normalize_category(category),
                    "mood": normalize_mood(mood),
                    "bbox": json.dumps(bbox, ensure_ascii=False) if bbox else "",
                }
            )

    dataframe = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"metadata 생성 완료: {output_path} ({len(dataframe)}개 항목)")
    return dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Hub 라벨에서 MoodFit metadata.csv 생성")
    parser.add_argument("--json-dir", default="recommend_pool/raw_data/labels")
    parser.add_argument("--image-dir", default="recommend_pool/raw_data/images")
    parser.add_argument("--output", default="recommend_pool/metadata.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_aihub_data(args.json_dir, args.image_dir, args.output)


if __name__ == "__main__":
    main()
