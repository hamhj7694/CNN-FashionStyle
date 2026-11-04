"""metadata.csv의 Lazy Crop 결과를 Cloudflare R2에 업로드한다.

실행 예시(PowerShell):
    $env:R2_ACCOUNT_ID = "..."
    $env:R2_ACCESS_KEY_ID = "..."
    $env:R2_SECRET_ACCESS_KEY = "..."
    $env:R2_BUCKET_NAME = "moodfit-assets"
    $env:R2_PUBLIC_BASE_URL = "https://assets.example.com"
    python upload_crops_to_r2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from curation import lazy_crop_item, lazy_thumbnail_item
from r2_storage import create_r2_client, get_r2_settings, upload_file_to_r2


ROOT = Path(__file__).resolve().parent


def upload_metadata_crops(
    metadata_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """크롭을 생성하고 R2 URL이 포함된 metadata_r2.csv를 만든다."""

    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    dataframe = pd.read_csv(metadata_path).fillna("")
    required_columns = {"id", "original_img_path", "bbox"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"metadata에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    dataframe["image_url"] = dataframe.get("image_url", "")
    dataframe["full_image_url"] = dataframe.get("full_image_url", "")
    settings = None if dry_run else get_r2_settings()
    client = None if dry_run else create_r2_client(settings)
    rows = dataframe.head(limit) if limit else dataframe
    uploaded_full_urls: dict[str, str] = {}

    for position, (index, row) in enumerate(rows.iterrows(), start=1):
        item = row.to_dict()
        full_path = lazy_thumbnail_item(item, cache_dir)
        bbox = str(item.get("bbox", "")).strip()
        crop_path = lazy_crop_item(item, cache_dir) if bbox and bbox not in {"[]", "{}"} else None
        if full_path is None and crop_path is None:
            print(f"[{position}] 건너뜀: 원본 이미지를 찾을 수 없음 (id={item.get('id')})")
            continue

        if dry_run:
            print(f"[{position}] dry-run: full={full_path}, crop={crop_path}")
            continue

        if full_path:
            if full_path not in uploaded_full_urls:
                full_key = f"full/{Path(full_path).name}"
                uploaded_full_urls[full_path] = upload_file_to_r2(full_path, full_key, client, settings)
            dataframe.loc[index, "full_image_url"] = uploaded_full_urls[full_path]

        if crop_path:
            crop_key = f"crops/crop_{item['id']}.jpg"
            dataframe.loc[index, "image_url"] = upload_file_to_r2(crop_path, crop_key, client, settings)
            print(f"[{position}] 업로드 완료: {crop_key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"metadata 저장 완료: {output_path}")
    return dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoodFit 크롭 이미지를 Cloudflare R2에 업로드")
    parser.add_argument("--metadata", default=str(ROOT / "recommend_pool" / "metadata.csv"))
    parser.add_argument("--output", default=str(ROOT / "recommend_pool" / "metadata_r2.csv"))
    parser.add_argument("--cache-dir", default=str(ROOT / "recommend_pool" / "processed_crops"))
    parser.add_argument("--limit", type=int, default=None, help="테스트용 업로드 개수")
    parser.add_argument("--dry-run", action="store_true", help="R2 업로드 없이 크롭/키만 확인")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    upload_metadata_crops(args.metadata, args.output, args.cache_dir, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
