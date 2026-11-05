"""MoodFit 추천 로직과 Lazy Cropping 캐시."""

from __future__ import annotations

import ast
from collections.abc import Collection
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

GENDER_FILTERS = {
    "남성": {"남성"},
    "여성": {"여성"},
    "공용": {"공용"},
    "전체": None,
    "여성 (공용 포함)": {"여성", "공용"},
    "남성 (공용 포함)": {"남성", "공용"},
}
ROOT = Path(__file__).resolve().parent


def _resolve_source_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT / path


def filter_by_gender(
    df_meta: pd.DataFrame, selected_gender: str | Collection[str] = "전체",
) -> pd.DataFrame:
    """선택된 성별들의 합집합으로 필터링한다. 빈 선택은 빈 결과를 반환한다."""

    if df_meta.empty:
        return df_meta.copy()
    # 기존 문자열 호출은 유지하고, UI에서는 독립된 성별 목록을 전달한다.
    if isinstance(selected_gender, str):
        allowed = GENDER_FILTERS.get(selected_gender)
        if allowed is None:
            return df_meta.copy()
    else:
        allowed = set(selected_gender) & {"남성", "여성", "공용"}
    if not allowed or "gender" not in df_meta.columns:
        return df_meta.iloc[0:0].copy()
    return df_meta[df_meta["gender"].isin(allowed)].copy()


def filter_catalog(
    df_meta: pd.DataFrame,
    selected_gender: str | Collection[str] = "전체",
    selected_category: str = "전체",
    selected_mood: str = "전체",
) -> pd.DataFrame:
    """성별 → 카테고리 → 무드 순서로 추천 후보를 필터링한다."""

    filtered = filter_by_gender(df_meta, selected_gender)
    if selected_category != "전체" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == selected_category].copy()
    if selected_mood != "전체" and "mood" in filtered.columns:
        filtered = filtered[filtered["mood"] == selected_mood].copy()
    return filtered


def _parse_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        parsed = value if isinstance(value, (list, tuple)) else json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        x1, y1, x2, y2 = (max(0, int(float(number))) for number in parsed)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _safe_cache_name(item_id: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", str(item_id)).strip("_") or "item"


def lazy_crop_item(
    item: dict[str, Any] | pd.Series,
    cache_dir: str | Path = "recommend_pool/processed_crops",
    size: tuple[int, int] = (512, 512),
) -> str | None:
    """추천 카드에 필요한 아이템만 최초 1회 크롭하고 캐시 경로를 반환한다."""

    record = dict(item)
    item_id = _safe_cache_name(record.get("id", "item"))
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_path = cache_root / f"crop_{item_id}.jpg"
    if os.path.exists(cached_path):
        return str(cached_path)

    source_path = _resolve_source_path(record.get("original_img_path", record.get("image_path", "")))
    if not source_path.is_file():
        return None

    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            bbox = _parse_bbox(record.get("bbox"))
            if bbox:
                x1, y1, x2, y2 = bbox
                x1, x2 = min(x1, image.width), min(x2, image.width)
                y1, y2 = min(y1, image.height), min(y2, image.height)
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            image = image.resize(size, Image.Resampling.LANCZOS)
            image.save(cached_path, format="JPEG", quality=91, optimize=True)
    except (OSError, ValueError):
        return None
    return str(cached_path) if cached_path.is_file() else None


def lazy_thumbnail_item(
    item: dict[str, Any] | pd.Series,
    cache_dir: str | Path = "recommend_pool/processed_crops",
    max_size: tuple[int, int] = (1200, 1200),
) -> str | None:
    """전신 원본을 비율 유지 축소해 웹용 썸네일로 캐시한다.

    같은 전신 이미지에 여러 바운딩 박스가 있을 수 있으므로 item id가 아니라
    원본 경로 해시를 캐시 키로 사용해 중복 썸네일 생성을 피한다.
    """

    record = dict(item)
    source_path = _resolve_source_path(record.get("original_img_path", record.get("image_path", "")))
    if not source_path.is_file():
        return None

    source_key = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_path = cache_root / f"full_{source_key}.jpg"
    if os.path.exists(cached_path):
        return str(cached_path)

    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.save(cached_path, format="JPEG", quality=88, optimize=True)
    except (OSError, ValueError):
        return None
    return str(cached_path) if cached_path.is_file() else None


def _sample_records(dataframe: pd.DataFrame, count: int, random_state: int | None) -> list[dict[str, Any]]:
    if dataframe.empty or count <= 0:
        return []
    return dataframe.sample(n=min(count, len(dataframe)), random_state=random_state).to_dict("records")


def get_recommendation_sections(
    user_embedding: Any,
    selected_category: str,
    df_meta: pd.DataFrame,
    item_embeddings: Any = None,
    selected_gender: str | Collection[str] = "전체",
    selected_mood: str = "전체",
    similar_count: int = 5,
    discovery_count: int = 5,
    random_state: int | None = 42,
) -> dict[str, list[dict[str, Any]]]:
    """유사 추천과 디스커버리 추천을 분리해 반환한다."""

    # Reset the index so every retained row maps to its original embedding position,
    # including catalogs with duplicate IDs or non-unique/non-sequential indexes.
    catalog = df_meta.reset_index(drop=True)
    filtered = filter_catalog(catalog, selected_gender, selected_category, selected_mood)
    if filtered.empty:
        return {"similar": [], "discovery": []}
    if "id" in filtered.columns:
        filtered = filtered.drop_duplicates(subset=["id"], keep="first")

    similar: list[dict[str, Any]] = []
    if user_embedding is not None and item_embeddings is not None:
        try:
            matrix = np.asarray(item_embeddings)
            valid_positions = filtered.index.to_numpy(dtype=np.intp)
            if matrix.ndim == 2 and len(matrix) == len(df_meta) and len(valid_positions):
                scores = cosine_similarity(np.asarray(user_embedding).reshape(1, -1), matrix[valid_positions])[0]
                ranked_positions = np.argsort(-scores)[:similar_count]
                similar = filtered.iloc[ranked_positions].copy().assign(
                    similarity=[float(scores[position]) for position in ranked_positions]
                ).to_dict("records")
        except (TypeError, ValueError):
            similar = []

    if not similar:
        similar = _sample_records(filtered, similar_count, random_state)

    similar_ids = {str(item.get("id")) for item in similar}
    remaining = filtered[~filtered["id"].astype(str).isin(similar_ids)] if "id" in filtered else filtered
    # 후보가 10개보다 적으면 같은 아이템을 다시 보여주지 않고 가능한 만큼만 반환한다.
    discovery = _sample_records(remaining, discovery_count, random_state)
    return {"similar": similar[:similar_count], "discovery": discovery[:discovery_count]}


def get_recommendations(
    user_embedding: Any,
    selected_category: str,
    df_meta: pd.DataFrame,
    item_embeddings: Any = None,
    selected_gender: str | Collection[str] = "전체",
    selected_mood: str = "전체",
) -> list[dict[str, Any]]:
    """기존 호출부 호환용: 유사 추천 5개와 디스커버리 5개를 합쳐 반환한다."""

    sections = get_recommendation_sections(
        user_embedding,
        selected_category,
        df_meta,
        item_embeddings,
        selected_gender,
        selected_mood,
    )
    return sections["similar"] + sections["discovery"]
