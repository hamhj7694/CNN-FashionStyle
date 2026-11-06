"""MoodFit AI Streamlit MVP."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from curation import (
    filter_catalog,
    get_recommendation_sections,
    lazy_crop_item,
    lazy_thumbnail_item,
)
from model import (
    EMBEDDING_DIM,
    EMBEDDING_VERSION,
    analyze_image,
    catalog_fingerprint,
    get_model_signature,
    load_feature_extractor,
)

try:
    import streamlit_shadcn_ui as ui

    HAS_SHADCN = True
except ImportError:
    ui = None
    HAS_SHADCN = False


ROOT = Path(__file__).resolve().parent
METADATA_PATHS = [
    ROOT / "recommend_pool" / "metadata_web.csv",
    ROOT / "recommend_pool" / "metadata_r2.csv",
    ROOT / "recommend_pool" / "metadata.csv",
    ROOT / "metadata.csv",
]
CROP_DIR = ROOT / "recommend_pool" / "processed_crops"

st.set_page_config(
    page_title="MoodFit AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink:#0F172A; --paper:#F8FAFC; --muted:#64748B; --line:#E2E8F0; --accent:#A78BFA; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid='stHeader'] { background:rgba(248,250,252,.86); }
    .block-container { max-width:1440px; padding:1.5rem 3rem 4rem; }
    .mf-nav { display:flex; align-items:center; justify-content:space-between; padding:.6rem 0 1.4rem; border-bottom:1px solid var(--line); margin-bottom:2.4rem; }
    .mf-logo { font-family:'Playfair Display',Georgia,serif; font-size:1.65rem; letter-spacing:-.03em; }
    .mf-logo span { color:var(--accent); }
    .mf-nav-note { color:var(--muted); font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
    .mf-kicker { color:#7C3AED; font-size:.74rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.6rem; }
    .mf-title { font-family:'Playfair Display',Georgia,serif; font-size:clamp(2.2rem,4vw,4.3rem); line-height:1.02; letter-spacing:-.04em; margin:0; }
    .mf-lede { color:var(--muted); max-width:560px; font-size:1rem; line-height:1.7; margin-top:1rem; }
    .mf-panel { background:white; border:1px solid var(--line); border-radius:22px; padding:1.25rem; min-height:365px; box-shadow:0 18px 50px rgba(15,23,42,.04); }
    .mf-panel-title { font-size:.76rem; font-weight:700; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; margin-bottom:.9rem; }
    .mf-analysis { min-height:300px; display:flex; flex-direction:column; justify-content:center; padding:1rem; }
    .mf-analysis h3 { font-family:'Playfair Display',Georgia,serif; font-size:2rem; margin:.9rem 0 .4rem; }
    .mf-analysis p { color:var(--muted); line-height:1.6; margin-bottom:1.5rem; }
    .mf-stat { display:flex; justify-content:space-between; border-top:1px solid var(--line); padding:.8rem 0; font-size:.9rem; }
    .mf-stat span:first-child { color:var(--muted); }
    .mf-pill { display:inline-block; background:#EDE9FE; color:#6D28D9; border-radius:999px; padding:.38rem .72rem; font-size:.74rem; font-weight:700; }
    .mf-pill-dark { background:var(--ink); color:white; }
    .mf-section { display:flex; align-items:end; justify-content:space-between; margin:3.2rem 0 1rem; }
    .mf-section h2 { font-family:'Playfair Display',Georgia,serif; font-size:2rem; margin:0; letter-spacing:-.03em; }
    .mf-section p { color:var(--muted); margin:.35rem 0 0; font-size:.9rem; }
    .mf-card-copy { padding:.8rem .1rem .2rem; }
    .mf-card-meta { color:var(--muted); font-size:.72rem; display:flex; justify-content:space-between; gap:.3rem; }
    .mf-card-title { font-weight:700; margin:.35rem 0 .2rem; font-size:.92rem; }
    .mf-image-link { display:block; color:inherit; text-decoration:none; cursor:zoom-in; }
    .mf-image-frame { position:relative; width:100%; overflow:hidden; border-radius:16px; background:#E2E8F0; aspect-ratio:4/5; }
    .stApp .mf-image-frame > img { position:absolute; inset:0; display:block; width:100% !important; height:100% !important; max-width:none; max-height:none; object-fit:cover; object-position:center; transition:transform .28s ease; }
    .mf-image-frame:hover img { transform:scale(1.045); }
    .mf-detail-image { border-radius:18px; overflow:hidden; background:#F1F5F9; }
    .mf-detail-image img { display:block; width:100%; max-height:420px; object-fit:contain; }
    .mf-crop-tile { border:1px solid var(--line); border-radius:14px; padding:.35rem; background:white; }
    .mf-empty { border:1px dashed #CBD5E1; border-radius:18px; padding:2rem; color:var(--muted); text-align:center; background:white; }
    .mf-preview-empty { min-height:260px; border:1px solid var(--line); border-radius:16px; padding:2rem; color:var(--muted); text-align:center; background:#FFFFFF; display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .mf-preview-empty strong { color:var(--ink); font-size:1rem; margin-bottom:.35rem; }
    .mf-preview-empty small { color:#94A3B8; font-size:.78rem; }
    div[data-testid='stFileUploaderDropzone'] { border:1px dashed #94A3B8; background:#F8FAFC; border-radius:16px; min-height:260px; }
    div[data-testid='stFileUploaderDropzone']:hover { border-color:#8B5CF6; background:#F5F3FF; }
    div[data-testid='stRadio'] > div { gap:.5rem; }
    div[data-testid='stRadio'] label { background:white; border:1px solid var(--line); border-radius:999px; padding:.32rem .8rem; }
    .stButton > button { border-radius:999px; border:1px solid var(--line); background:white; color:var(--ink); }
    .stButton > button:hover { border-color:#8B5CF6; color:#6D28D9; }

    /* Keep image previews bounded while the workspace follows its content. */
    .st-key-look_workspace { --mf-control-height:88px; height:auto; max-height:none; overflow:visible; padding-right:0; }
    .mf-nav { gap:.75rem; flex-wrap:wrap; }
    .st-key-look_workspace .mf-title { font-size:clamp(2rem,3.5vw,3.4rem); }
    .st-key-look_workspace .mf-analysis { min-height:0; padding:.6rem; }
    .st-key-look_workspace .mf-analysis h3 { margin:.45rem 0; }
    .st-key-look_workspace .mf-analysis p { margin-bottom:.7rem; }
    .mf-upload-frame { height:850px; max-height:850px; position:relative; overflow:hidden; border-radius:16px; background:#E2E8F0; }
    .mf-selected-frame { height:850px; max-height:850px; position:relative; overflow:hidden; border-radius:16px; background:#E2E8F0; }
    .stApp .mf-upload-frame > img, .stApp .mf-selected-frame > img { position:absolute; inset:0; width:100% !important; height:100% !important; max-height:none; object-fit:contain; object-position:center; }
    .mf-selected-caption { margin:.4rem 0 0; color:var(--muted); font-size:.8rem; }
    .st-key-upload_control [data-testid="stFileUploaderDropzone"] { box-sizing:border-box; height:var(--mf-control-height); min-height:var(--mf-control-height); padding:.75rem 1rem; border-radius:16px; }
    .mf-selected-empty { box-sizing:border-box; height:var(--mf-control-height); min-height:var(--mf-control-height); padding:.75rem 1rem; border:1px solid var(--line); border-radius:16px; background:#FFFFFF; display:flex; flex-direction:column; justify-content:center; }
    .mf-selected-empty .mf-panel-title { margin-bottom:.35rem; }
    .mf-selected-empty p { margin:0; color:var(--muted); font-size:.85rem; line-height:1.6; }

    /* The native button covers the image without navigating away from the session. */
    [class*="st-key-recommend_pick_"] { position:relative; }
    [class*="st-key-recommend_pick_"] [data-testid="stElementContainer"]:has(> [data-testid="stButton"]) { position:absolute; inset:0; width:100%; height:100%; z-index:1; }
    [class*="st-key-recommend_pick_"] [data-testid="stButton"] { width:100%; height:100%; }
    [class*="st-key-recommend_pick_"] [data-testid="stButton"] > button { width:100%; height:100%; padding:0; border:0; border-radius:16px; background:transparent !important; color:transparent !important; cursor:pointer; }
    [class*="st-key-recommend_pick_"] [data-testid="stButton"] > button:focus-visible { outline:3px solid #7C3AED; outline-offset:-3px; }
    [class*="st-key-recommend_pick_"]:hover img { transform:scale(1.045); }
    .mf-card-selected { outline:3px solid #7C3AED; outline-offset:-3px; }
    .st-key-selected_image_viewer [data-testid="stElementContainer"], .st-key-upload_image_viewer [data-testid="stElementContainer"] { height:850px !important; }
    .st-key-selected_image_viewer iframe, .st-key-upload_image_viewer iframe { height:100% !important; width:100%; }
    @media (max-width:900px) {
        .mf-upload-frame, .mf-selected-frame { height:700px; }
        .st-key-selected_image_viewer [data-testid="stElementContainer"], .st-key-upload_image_viewer [data-testid="stElementContainer"] { height:700px !important; }
    }
    @media (max-width:640px) {
        .st-key-look_workspace { --mf-control-height:104px; }
        .mf-upload-frame { height:420px; }
        .mf-selected-frame { height:420px; }
        .st-key-selected_image_viewer [data-testid="stElementContainer"], .st-key-upload_image_viewer [data-testid="stElementContainer"] { height:420px !important; }
    }
    .st-key-catalog_grid [data-testid="stHorizontalBlock"] { display:grid !important; grid-template-columns:repeat(8,minmax(0,1fr)); gap:12px !important; }
    .st-key-catalog_grid [data-testid="stColumn"] { width:100% !important; min-width:0 !important; flex:unset !important; }
    .st-key-catalog_split .st-key-catalog_grid [data-testid="stHorizontalBlock"] { grid-template-columns:repeat(4,minmax(0,1fr)); }
    .st-key-catalog_split > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { align-self:stretch !important; }
    .st-key-catalog_preview { position:sticky; top:4rem; }
    .st-key-catalog_preview_viewer [data-testid="stElementContainer"] { height:clamp(320px,70vh,760px) !important; }
    .st-key-catalog_preview_viewer iframe { height:100% !important; width:100%; }
    .mf-gallery-preview-mood { margin:.6rem 0; color:var(--ink); font-size:1.05rem; }
    @media (max-width:640px) {
        .st-key-catalog_split .st-key-catalog_grid [data-testid="stHorizontalBlock"] { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .st-key-catalog_preview { position:static; }
    }
    .mf-gallery-copy { color:var(--muted); font-size:.74rem; line-height:1.5; margin:.4rem 0 .9rem; overflow-wrap:anywhere; }
    .mf-gallery-copy strong { color:var(--ink); font-weight:500; }
    .mf-gallery-heading h2 { margin:0; font-family:'Playfair Display',Georgia,serif; font-size:2rem; }
    .mf-gallery-heading p { margin:.5rem 0 1.2rem; color:var(--muted); }
    @media (max-width:900px) { .st-key-catalog_grid [data-testid="stHorizontalBlock"] { grid-template-columns:repeat(4,minmax(0,1fr)); } }
    @media (max-width:640px) { .st-key-catalog_grid [data-testid="stHorizontalBlock"] { grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


@st.cache_data(show_spinner=False)
def load_metadata(file_signature: tuple[Any, ...] = ()) -> tuple[pd.DataFrame, bool]:
    for path in METADATA_PATHS:
        if path.is_file():
            return _normalize_metadata(pd.read_csv(path)), False

    legacy_roots = [
        ROOT / "CNN_train_image",
        ROOT / "K-Fashion",
        ROOT / "K-Fashion 이미지",
        ROOT / "새 폴더" / "CNN_train_test_image",
    ]
    legacy_root = next((path for path in legacy_roots if path.is_dir()), None)
    if legacy_root:
        rows: list[dict[str, Any]] = []
        for image_path in sorted(legacy_root.rglob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rows.append(
                {
                    "id": f"demo_{len(rows) + 1}",
                    "original_img_path": str(image_path),
                    "original_json_path": "",
                    "gender": "공용",
                    "category": "기타",
                    "mood": image_path.parent.name or "미분류",
                    "bbox": "",
                    "image_url": "",
                    "full_image_url": "",
                    "image_path": "",
                    "full_image_path": "",
                    "crop_image_path": "",
                }
            )
            if len(rows) >= 60:
                break
        if rows:
            return _normalize_metadata(pd.DataFrame(rows)), True

    # 정식 metadata CSV가 없어도 준비된 로컬 추천 이미지로 데모 카탈로그를 구성한다.
    full_demo_paths = sorted(CROP_DIR.glob("full_*.jpg"))
    demo_paths = full_demo_paths if len(full_demo_paths) >= 10 else sorted(CROP_DIR.glob("*.jpg"))
    if demo_paths:
        rows = []
        for image_path in demo_paths:
            rows.append(
                {
                    "id": f"local_demo_{image_path.stem}",
                    "original_img_path": str(image_path),
                    "original_json_path": "",
                    "gender": "공용",
                    "category": "추천 아이템",
                    "mood": "",
                    "bbox": "",
                    "image_url": "",
                    "full_image_url": "",
                    "image_path": str(image_path),
                    "full_image_path": str(image_path),
                    "crop_image_path": str(image_path),
                }
            )
        return _normalize_metadata(pd.DataFrame(rows)), True
    return _normalize_metadata(pd.DataFrame()), False


def _normalize_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    if "image_path" in dataframe.columns:
        if "original_img_path" not in dataframe.columns:
            dataframe = dataframe.rename(columns={"image_path": "original_img_path"})
        else:
            dataframe = dataframe.drop(columns=["image_path"])
    required = [
        "id", "original_img_path", "original_json_path", "gender", "category", "mood", "bbox",
        "image_url", "full_image_url", "full_image_path", "crop_image_path",
    ]
    for column in required:
        if column not in dataframe.columns:
            dataframe[column] = ""
    dataframe = dataframe[required].fillna("")
    for column in ("mood", "category"):
        dataframe[column] = dataframe[column].astype(str).str.strip()
    return dataframe


@st.cache_data(show_spinner=False)
def _predict_demo_catalog_moods(
    image_paths: tuple[str, ...], checkpoint_path: str, checkpoint_sha256: str,
) -> dict[str, str]:
    """메타데이터가 없는 데모 추천 이미지에 현재 모델의 무드 라벨을 붙인다."""
    network = load_feature_extractor(checkpoint_path, device="cpu")
    predictions: dict[str, str] = {}
    for image_path in image_paths:
        try:
            predictions[image_path] = analyze_image(Path(image_path), network)["mood"]
        except (OSError, ValueError, RuntimeError):
            predictions[image_path] = ""
    return predictions


def _attach_demo_moods(dataframe: pd.DataFrame, model: Any) -> pd.DataFrame:
    """데모 카탈로그에 실제 모델 예측 무드를 추가한다."""
    if dataframe.empty or "mood" not in dataframe.columns:
        return dataframe
    if dataframe["mood"].astype(str).str.strip().ne("").any():
        return dataframe
    image_paths = tuple(str(value) for value in dataframe["original_img_path"])
    predictions = _predict_demo_catalog_moods(
        image_paths,
        str(model.checkpoint_path),
        model.checkpoint_sha256,
    )
    labeled = dataframe.copy()
    labeled["mood"] = [predictions.get(path, "") for path in image_paths]
    return labeled


def _pick_mood_recommendations(
    dataframe: pd.DataFrame, mood_scores: dict[str, float], count: int = 5,
    random_state: int | None = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Top-1 무드 5개와 Top-2/3 무드 5개를 랜덤하게 선택한다."""
    ranked_moods = [
        mood for mood, _ in sorted(mood_scores.items(), key=lambda item: item[1], reverse=True)
    ]
    if dataframe.empty or not ranked_moods:
        return {"similar": [], "discovery": []}

    if "id" in dataframe.columns:
        dataframe = dataframe.drop_duplicates(subset=["id"], keep="first")
    rng = np.random.default_rng(random_state)

    def sample_pool(pool: pd.DataFrame, amount: int, used_ids: set[str]) -> list[dict[str, Any]]:
        if pool.empty or amount <= 0:
            return []
        if "id" in pool.columns:
            pool = pool[~pool["id"].astype(str).isin(used_ids)]
        if pool.empty:
            return []
        sampled = pool.sample(n=min(amount, len(pool)), random_state=rng)
        return sampled.to_dict("records")

    used_ids: set[str] = set()
    top1_pool = dataframe[dataframe["mood"] == ranked_moods[0]]
    top1 = sample_pool(top1_pool, count, used_ids)
    used_ids.update(str(item.get("id")) for item in top1)

    top23_moods = set(ranked_moods[1:3])
    top23_pool = dataframe[dataframe["mood"].isin(top23_moods)]
    top23 = sample_pool(top23_pool, count, used_ids)
    used_ids.update(str(item.get("id")) for item in top23)

    return {"similar": top1[:count], "discovery": top23[:count]}


def _first_source(item: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = str(item.get(field, "")).strip()
        if value.startswith(("http://", "https://")) or Path(value).is_file():
            return value
    return ""


def _as_image_src(source: str) -> str:
    if source.startswith(("http://", "https://", "data:")):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_image_hover(source: str, alt: str, query: str = "") -> None:
    image_src = _as_image_src(source)
    if not image_src:
        st.markdown('<div class="mf-empty">이미지 준비 중</div>', unsafe_allow_html=True)
        return
    image_html = f'<div class="mf-image-frame"><img src="{html.escape(image_src, quote=True)}" alt="{html.escape(alt, quote=True)}"></div>'
    if query:
        image_html = f'<a class="mf-image-link" href="?{html.escape(query, quote=True)}">{image_html}</a>'
    st.markdown(image_html, unsafe_allow_html=True)


def render_badge(label: str, key: str, dark: bool = False) -> None:
    if HAS_SHADCN and hasattr(ui, "badge"):
        try:
            ui.badge(label, key=key)
            return
        except Exception:
            pass
    class_name = "mf-pill mf-pill-dark" if dark else "mf-pill"
    st.markdown(f'<span class="{class_name}">{html.escape(label)}</span>', unsafe_allow_html=True)


def _full_source(item: dict[str, Any]) -> str:
    source = _first_source(item, ("full_image_url", "full_image_path"))
    if source:
        return source
    generated = lazy_thumbnail_item(item, CROP_DIR)
    if generated:
        return generated
    return _first_source(item, ("image_url", "crop_image_path"))


def _crop_source(item: dict[str, Any]) -> str:
    source = _first_source(item, ("image_url", "crop_image_path"))
    bbox = str(item.get("bbox", "")).strip()
    # bbox가 없는 데모/전신 row를 다시 crop으로 보여주면 같은 이미지가 중복된다.
    if source and bbox and bbox not in {"[]", "{}"}:
        return source
    if not bbox or bbox in {"[]", "{}"}:
        return ""
    generated = lazy_crop_item(item, CROP_DIR)
    return generated or ""


def _select_item(item: dict[str, Any]) -> None:
    # Button callbacks run before the next script execution; keep the upload intact.
    st.session_state["selected_item"] = item
    st.session_state["detail_mode"] = "full"
    _clear_query_params()


def ensure_selected_recommendation(
    recommendations: list[dict[str, Any]], dataframe: pd.DataFrame,
) -> dict[str, Any] | None:
    selected = st.session_state.get("selected_item")
    if selected and "id" in dataframe.columns:
        if not dataframe["id"].astype(str).eq(str(selected.get("id", ""))).any():
            selected = None
    if selected is None and recommendations:
        selected = recommendations[0]
    st.session_state["selected_item"] = selected
    return selected


def render_bounded_image(source: str, alt: str, frame_class: str) -> None:
    image_src = _as_image_src(source)
    if not image_src:
        st.markdown('<div class="mf-empty">이미지 준비 중</div>', unsafe_allow_html=True)
        return
    st.markdown(
        f'<div class="{frame_class}"><img src="{html.escape(image_src, quote=True)}" '
        f'alt="{html.escape(alt, quote=True)}"></div>',
        unsafe_allow_html=True,
    )


def image_viewer_html(image_src: str, alt: str, frame_class: str = "mf-selected-frame") -> str:
    """Build the isolated viewer using escaped image attributes and local JS only."""
    template = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html, body { margin:0; width:100%; height:100%; overflow:hidden; font-family:system-ui,sans-serif; }
.mf-selected-frame { position:relative; width:100%; height:100%; max-height:100%; overflow:hidden; border-radius:16px; background:#E2E8F0; cursor:grab; touch-action:none; user-select:none; outline-offset:-3px; }
.mf-selected-frame:focus-visible { outline:3px solid #7C3AED; }
.mf-selected-frame.dragging { cursor:grabbing; }
.mf-selected-frame img { position:absolute; top:0; left:0; display:block; max-width:none; max-height:none; transform-origin:0 0; pointer-events:none; user-select:none; visibility:hidden; will-change:transform; }
.viewer-controls { position:absolute; top:10px; right:10px; display:flex; align-items:center; gap:6px; z-index:1; }
.viewer-controls button, .viewer-controls output { border:0; border-radius:8px; padding:6px 9px; background:rgba(255,255,255,.93); color:#0F172A; font-size:12px; }
.viewer-controls button { cursor:pointer; }
.viewer-controls button:focus-visible { outline:2px solid #7C3AED; }
.viewer-hint { position:absolute; bottom:10px; left:10px; padding:5px 8px; border-radius:8px; background:rgba(15,23,42,.7); color:white; font-size:11px; pointer-events:none; }
.viewer-status { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#64748B; padding:20px; text-align:center; pointer-events:none; }
.viewer-status[hidden] { display:none; }
</style>
</head>
<body>
<div id="viewer" class="mf-selected-frame" tabindex="0" aria-label="선택된 추천 이미지. 휠로 확대 축소, 드래그로 이동, 더블클릭으로 초기화">
  <img id="image" src="__IMAGE_SRC__" alt="__IMAGE_ALT__" draggable="false">
  <div class="viewer-controls">
    <output id="zoom" aria-label="확대 비율">100%</output>
    <button id="reset" type="button" aria-label="이미지 확대와 위치 초기화">초기화</button>
  </div>
  <div class="viewer-hint">휠: 확대·축소 · 드래그: 이동 · 더블클릭: 초기화</div>
  <div id="status" class="viewer-status" role="status">이미지를 불러오는 중입니다.</div>
</div>
<script>
(() => {
  'use strict';
  const frame = document.getElementById('viewer');
  const image = document.getElementById('image');
  const zoomLabel = document.getElementById('zoom');
  const resetButton = document.getElementById('reset');
  const status = document.getElementById('status');
  let ready = false, zoom = 1, base = 1, x = 0, y = 0;
  let width = 0, height = 0, pointer = null, lastX = 0, lastY = 0;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  function paint() {
    if (!ready) return;
    const scale = base * zoom;
    // Center a zoomed-out image on each axis that is smaller than the frame.
    const renderedWidth = image.naturalWidth * scale;
    const renderedHeight = image.naturalHeight * scale;
    x = renderedWidth <= width ? (width - renderedWidth) / 2 : clamp(x, width - renderedWidth, 0);
    y = renderedHeight <= height ? (height - renderedHeight) / 2 : clamp(y, height - renderedHeight, 0);
    image.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  }

  function reset() {
    if (!image.naturalWidth || !image.naturalHeight) return;
    const bounds = frame.getBoundingClientRect();
    width = bounds.width;
    height = bounds.height;
    if (!width || !height) return;
    // Cover fit: keep the existing 100% startup and reset view.
    base = Math.max(width / image.naturalWidth, height / image.naturalHeight);
    zoom = 1;
    x = (width - image.naturalWidth * base) / 2;
    y = (height - image.naturalHeight * base) / 2;
    image.style.width = `${image.naturalWidth}px`;
    image.style.height = `${image.naturalHeight}px`;
    ready = true;
    image.style.visibility = 'visible';
    status.hidden = true;
    paint();
  }

  function zoomAt(nextZoom, px, py) {
    if (!ready) return;
    nextZoom = clamp(nextZoom, 0.25, 6);
    const ratio = nextZoom / zoom;
    // Keep the image point underneath the cursor stationary while zooming.
    x = px - (px - x) * ratio;
    y = py - (py - y) * ratio;
    zoom = nextZoom;
    paint();
  }

  frame.addEventListener('wheel', event => {
    if (!ready) return;
    event.preventDefault();
    const bounds = frame.getBoundingClientRect();
    const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1;
    const delta = clamp(event.deltaY * units, -500, 500);
    zoomAt(zoom * Math.exp(-delta * 0.002), event.clientX - bounds.left, event.clientY - bounds.top);
  }, { passive: false });

  frame.addEventListener('pointerdown', event => {
    if (!ready || event.button !== 0 || pointer !== null || event.target.closest('button')) return;
    event.preventDefault();
    frame.focus({ preventScroll: true });
    pointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    frame.setPointerCapture(pointer);
    frame.classList.add('dragging');
  });
  frame.addEventListener('pointermove', event => {
    if (event.pointerId !== pointer) return;
    x += event.clientX - lastX;
    y += event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    paint();
  });
  function endDrag(event) {
    if (event.pointerId !== pointer) return;
    const previous = pointer;
    pointer = null;
    frame.classList.remove('dragging');
    if (frame.hasPointerCapture(previous)) frame.releasePointerCapture(previous);
  }
  frame.addEventListener('pointerup', endDrag);
  frame.addEventListener('pointercancel', endDrag);
  frame.addEventListener('lostpointercapture', endDrag);
  frame.addEventListener('dblclick', reset);
  resetButton.addEventListener('click', reset);
  frame.addEventListener('keydown', event => {
    if (!ready || event.target.closest('button')) return;
    switch (event.key) {
      case '+': case '=': zoomAt(zoom * 1.2, width / 2, height / 2); break;
      case '-': zoomAt(zoom / 1.2, width / 2, height / 2); break;
      case '0': reset(); break;
      case 'ArrowLeft': x += 35; paint(); break;
      case 'ArrowRight': x -= 35; paint(); break;
      case 'ArrowUp': y += 35; paint(); break;
      case 'ArrowDown': y -= 35; paint(); break;
      default: return;
    }
    event.preventDefault();
  });
  image.addEventListener('load', reset);
  image.addEventListener('error', () => {
    ready = false;
    image.style.visibility = 'hidden';
    status.hidden = false;
    status.textContent = '이미지를 불러오지 못했습니다. 이미지를 다시 선택해 주세요.';
  });
  new ResizeObserver(() => {
    const bounds = frame.getBoundingClientRect();
    if (bounds.width !== width || bounds.height !== height) reset();
  }).observe(frame);
  if (image.complete && image.naturalWidth) reset();
})();
</script>
</body>
</html>
"""
    if frame_class not in {"mf-selected-frame", "mf-upload-frame"}:
        raise ValueError("지원하지 않는 이미지 프레임입니다.")
    template = template.replace("mf-selected-frame", frame_class).replace(
        "선택된 추천 이미지. 휠로 확대 축소, 드래그로 이동, 더블클릭으로 초기화",
        html.escape(f"{alt}. 휠로 확대 축소, 드래그로 이동, 더블클릭으로 초기화", quote=True),
    )
    return template.replace("__IMAGE_SRC__", html.escape(image_src, quote=True)).replace(
        "__IMAGE_ALT__", html.escape(alt, quote=True),
    )


def render_interactive_image(source: str, *, viewer_key: str, alt: str, frame_class: str, height: int = 850) -> None:
    image_src = _as_image_src(source)
    if not image_src:
        st.info("이미지를 불러올 수 없습니다. 이미지를 다시 선택해 주세요.")
        return
    document = image_viewer_html(image_src, alt, frame_class)
    # Separate iframes keep upload/recommendation zoom and drag independent.
    with st.container(key=viewer_key):
        if hasattr(st, "iframe"):
            st.iframe(document, height=height, tab_index=0, alt=f"휠로 확대 축소하고 드래그로 이동하는 {alt}")
        else:
            from streamlit.components.v1 import html as component_html
            component_html(document, height=height, scrolling=False, tab_index=0)


def render_selected_recommendation(item: dict[str, Any] | None, show_title: bool = True) -> None:
    if item is None:
        st.markdown(
            '<div class="mf-selected-empty">'
            '<div class="mf-panel-title">선택된 추천 이미지</div>'
            '<p>아래 추천 이미지를 선택하면 이곳에 표시됩니다.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return
    if show_title:
        st.markdown('<div class="mf-panel-title">선택된 추천 이미지</div>', unsafe_allow_html=True)
    render_interactive_image(
        _full_source(item), viewer_key="selected_image_viewer",
        alt="선택된 추천 착장", frame_class="mf-selected-frame",
    )
    caption = str(item.get("mood", "미분류"))
    st.markdown(f'<p class="mf-selected-caption">{html.escape(caption)}</p>', unsafe_allow_html=True)

def render_card(item: dict[str, Any], section_key: str, index: int) -> None:
    full_source = _full_source(item)
    with st.container(key=f"recommend_pick_{section_key}_{index}"):
        selected = st.session_state.get("selected_item") or {}
        is_selected = str(selected.get("id", "")) == str(item.get("id", ""))
        image_src = _as_image_src(full_source)
        frame_class = "mf-image-frame mf-card-selected" if is_selected else "mf-image-frame"
        if image_src:
            st.markdown(
                f'<div class="{frame_class}"><img src="{html.escape(image_src, quote=True)}" '
                f'alt="{html.escape(str(item.get("category", "추천 착장")), quote=True)}"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="mf-image-frame mf-empty">이미지 준비 중</div>', unsafe_allow_html=True)
        st.button(
            f"{item.get('category', '추천 착장')} · {item.get('mood', '')} 이미지 선택",
            key=f"select_recommendation_{section_key}_{index}",
            on_click=_select_item, args=(item,), use_container_width=True,
        )

    mood = html.escape(str(item.get("mood", "미분류")))
    score = item.get("similarity")
    score_text = f"유사도 {float(score) * 100:.0f}%" if score is not None else "curated pick"
    if HAS_SHADCN and hasattr(ui, "card"):
        try:
            ui.card(
                title=mood,
                content=f"{score_text} · MoodFit edit",
                key=f"card_{section_key}_{index}",
            )
        except Exception:
            pass
    else:
        st.markdown(
            f'<div class="mf-card-copy"><div class="mf-card-meta"><span>{score_text}</span></div><div class="mf-card-title">{mood}</div></div>',
            unsafe_allow_html=True,
        )

@st.cache_resource(show_spinner=False)
def load_model_safely(signature: tuple[str, int, int]) -> tuple[Any, str]:
    try:
        return load_feature_extractor(signature[0]), ""
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _load_embedding_file(
    path: str, file_signature: tuple[int, ...], rows: int, model_id: str, catalog_id: str,
) -> tuple[np.ndarray | None, str]:
    # file_signature는 벡터 또는 manifest가 바뀌면 캐시를 갱신하기 위한 인자다.
    try:
        index_path = Path(path)
        manifest = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("추천 인덱스 정보의 형식이 올바르지 않습니다.")
        if (manifest.get("embedding_version") != EMBEDDING_VERSION
                or manifest.get("checkpoint_sha256") != model_id
                or manifest.get("catalog_fingerprint") != catalog_id):
            raise ValueError("추천 후보 정보가 변경되었습니다. 후보 인덱스를 다시 생성해 주세요.")
        expected_checksum = manifest.get("embedding_sha256")
        if not expected_checksum:
            raise ValueError("추천 인덱스 검증 정보가 없습니다. 후보 인덱스를 다시 생성해 주세요.")
        digest = hashlib.sha256()
        with index_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_checksum:
            raise ValueError("추천 인덱스가 갱신 중이거나 손상되었습니다. 잠시 후 다시 확인해 주세요.")
        embeddings = np.load(index_path, allow_pickle=False)
        if (embeddings.shape != (rows, EMBEDDING_DIM)
                or manifest.get("shape") != list(embeddings.shape)
                or not np.isfinite(embeddings).all()):
            raise ValueError("추천 벡터의 개수 또는 차원이 맞지 않습니다.")
        return embeddings, ""
    except (OSError, ValueError, TypeError) as exc:
        return None, str(exc)


def load_item_embeddings(dataframe: pd.DataFrame, model: Any) -> tuple[np.ndarray | None, str]:
    if model is None:
        return None, ""
    errors = []
    fingerprint = catalog_fingerprint(dataframe.to_dict("records"))
    for path in (ROOT / "recommend_pool" / "item_embeddings_web.npy", ROOT / "recommend_pool" / "item_embeddings.npy"):
        manifest_path = path.with_suffix(".json")
        if not path.is_file() or not manifest_path.is_file():
            continue
        try:
            signature = tuple(value for file in (path, manifest_path)
                              for value in (file.stat().st_mtime_ns, file.stat().st_size))
            embeddings, error = _load_embedding_file(
                str(path), signature, len(dataframe), model.checkpoint_sha256, fingerprint,
            )
            if embeddings is not None:
                return embeddings, ""
            errors.append(error)
        except OSError as exc:
            errors.append(str(exc))
    return None, errors[0] if errors else "유사 추천용 후보 인덱스가 아직 없습니다."


def analyze_upload(uploaded_file: Any, model: Any) -> tuple[dict[str, Any] | None, str]:
    if uploaded_file is None or model is None:
        return None, ""
    payload = uploaded_file.getvalue()
    key = (hashlib.sha256(payload).hexdigest(), model.checkpoint_sha256)
    cached = st.session_state.get("upload_analysis")
    if cached and cached["key"] == key:
        return cached["result"], ""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            result = analyze_image(image, model)
        st.session_state["upload_analysis"] = {"key": key, "result": result}
        return result, ""
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return None, f"이미지 분석을 완료하지 못했습니다: {exc}"


def _get_query_param(name: str) -> str:
    if hasattr(st, "query_params"):
        value = st.query_params.get(name, "")
        return str(value)
    values = st.experimental_get_query_params().get(name, [""])
    return str(values[0]) if values else ""


def _clear_query_params() -> None:
    if hasattr(st, "query_params"):
        st.query_params.clear()
    else:
        st.experimental_set_query_params()


def apply_query_selection(dataframe: pd.DataFrame) -> None:
    selected_id = _get_query_param("select_id")
    if not selected_id or dataframe.empty or "id" not in dataframe.columns:
        return
    matches = dataframe[dataframe["id"].astype(str) == selected_id]
    if matches.empty:
        return
    current_id = str((st.session_state.get("selected_item") or {}).get("id", ""))
    if current_id != selected_id:
        st.session_state["selected_item"] = matches.iloc[0].to_dict()
    st.session_state["detail_mode"] = "crop" if _get_query_param("view") == "crop" else "full"


def render_analysis_panel(
    uploaded_file: Any,
    analysis: dict[str, Any] | None = None, analysis_error: str = "",
) -> None:
    st.markdown('<div class="mf-panel-title">02 / MOOD DIAGNOSIS</div>', unsafe_allow_html=True)
    if uploaded_file and analysis:
        render_badge("MOOD DETECTED", "mood_detected", dark=True)
        mood = html.escape(analysis["mood"])
        alternatives = sorted(analysis["mood_scores"].items(), key=lambda item: item[1], reverse=True)[1:3]
        secondary = html.escape(" · ".join(f"{name} {score:.1%}" for name, score in alternatives))
        st.markdown(
            f'<div class="mf-analysis"><h3>{mood}</h3>'
            '<p>업로드한 착장에서 감지한 패션 무드입니다. <br>무드와 의류 분류는 아래 필터에서 선택해 보세요.</p>'
            f'<div class="mf-stat"><span>예측 점수</span><b>{analysis["confidence"]:.1%}</b></div>'
            f'<div class="mf-stat"><span>함께 감지된 무드</span><b>{secondary}</b></div></div>',
            unsafe_allow_html=True,
        )
    elif uploaded_file:
        render_badge("ANALYSIS PENDING", "mood_pending")
        st.info(analysis_error or "학습 완료 후 V4 최고 모델 체크포인트를 저장하면 무드 분석을 사용할 수 있습니다.")
        if st.button("모델 다시 확인", key="reload_mood_model"):
            load_model_safely.clear()
            st.session_state.pop("upload_analysis", None)
            _rerun()
    else:
        st.markdown('<div class="mf-analysis"><div style="font-size:2.5rem;">✦</div><h3>Start with a look</h3><p>당신과 어울리는 패션 취향을 확인해 보세요.<br>추천룩, 시밀러룩, 커플룩으로 활용해 보세요.</p></div>', unsafe_allow_html=True)



def catalog_filter_options(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        return ["전체"]
    values = dataframe[column].dropna().astype(str).str.strip()
    return ["전체"] + sorted({value for value in values if value and value != "전체"})


def reset_catalog_filters() -> None:
    st.session_state["mood_filter"] = "전체"
    st.session_state["category_filter"] = "전체"
    st.session_state["catalog_page"] = 1
    st.session_state["catalog_selected_item"] = None


def reset_catalog_page() -> None:
    st.session_state["catalog_page"] = 1
    st.session_state["catalog_selected_item"] = None


def open_catalog(mood: str | None = None, category: str | None = None) -> None:
    if mood is not None:
        st.session_state["mood_filter"] = mood
    if category is not None:
        st.session_state["category_filter"] = category
    reset_catalog_page()
    st.session_state["show_catalog"] = True


def close_catalog() -> None:
    st.session_state["show_catalog"] = False


def select_catalog_item(item: dict[str, Any]) -> None:
    _select_item(item)
    st.session_state["catalog_selected_item"] = item


def close_catalog_preview() -> None:
    st.session_state["catalog_selected_item"] = None


def move_catalog_page(offset: int) -> None:
    st.session_state["catalog_page"] = max(1, st.session_state.get("catalog_page", 1) + offset)


def catalog_image_page(
    dataframe: pd.DataFrame, mood: str, category: str, page: int = 1, page_size: int = 60,
) -> tuple[pd.DataFrame, int, int]:
    """Filter before deduplicating photos, then paginate a stable mixed order."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    images = filter_catalog(dataframe, "전체", category, mood).reset_index(drop=True).copy()
    image_keys = pd.Series("", index=images.index, dtype="object")
    for field in ("full_image_url", "full_image_path", "original_img_path", "image_url", "crop_image_path"):
        if field in images:
            values = images[field].fillna("").astype(str).str.strip()
            image_keys = image_keys.mask(image_keys.eq(""), values)
    fallback = images["id"].astype(str) if "id" in images else images.index.astype(str)
    images["_catalog_image_key"] = image_keys.mask(image_keys.eq(""), "item:" + pd.Series(fallback, index=images.index))
    images = images.drop_duplicates("_catalog_image_key").sample(frac=1, random_state=42)
    total = len(images)
    last_page = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, int(page)), last_page)
    start = (page - 1) * page_size
    return images.iloc[start:start + page_size].drop(columns="_catalog_image_key"), total, page


def render_catalog_filters(dataframe: pd.DataFrame) -> tuple[str, str]:
    mood_options = catalog_filter_options(dataframe, "mood")
    category_options = catalog_filter_options(dataframe, "category")
    for key, options in (("mood_filter", mood_options), ("category_filter", category_options)):
        if st.session_state.get(key, "전체") not in options:
            st.session_state[key] = "전체"
    filter_mood, filter_category, filter_actions = st.columns([1, 1, .5], gap="medium", vertical_alignment="bottom")
    with filter_mood:
        selected_mood = st.selectbox("무드", mood_options, key="mood_filter", disabled=dataframe.empty, on_change=reset_catalog_page)
    with filter_category:
        selected_category = st.selectbox("의류 분류", category_options, key="category_filter", disabled=dataframe.empty, on_change=reset_catalog_page)
    with filter_actions:
        st.button("필터 초기화", key="reset_catalog_filters", on_click=reset_catalog_filters, use_container_width=True, disabled=dataframe.empty)
    return selected_mood, selected_category


def render_catalog_card(item: dict[str, Any], index: int) -> None:
    category = html.escape(str(item.get("category", "기타")))
    mood = html.escape(str(item.get("mood", "미분류")))
    source = _as_image_src(_full_source(item))
    selected = st.session_state.get("catalog_selected_item") or {}
    frame_class = "mf-image-frame mf-card-selected" if str(selected.get("id", "")) == str(item.get("id", "")) else "mf-image-frame"
    with st.container(key=f"recommend_pick_catalog_{index}"):
        if source:
            st.markdown(
                f'<div class="{frame_class}"><img src="{html.escape(source, quote=True)}" '
                f'alt="{category} · {mood}" loading="lazy" decoding="async"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="mf-image-frame mf-empty">이미지 준비 중</div>', unsafe_allow_html=True)
        st.button(
            f"{item.get('category', '착장')} · {item.get('mood', '')} 크게 비교하기",
            key=f"select_catalog_{index}", on_click=select_catalog_item, args=(item,),
            use_container_width=True,
        )
    st.markdown(f'<p class="mf-gallery-copy"><strong>{mood}</strong></p>', unsafe_allow_html=True)


def render_catalog_grid(rows: pd.DataFrame, page: int, page_size: int, column_count: int) -> None:
    with st.container(key="catalog_grid"):
        items = rows.to_dict("records")
        for row_start in range(0, len(items), column_count):
            columns = st.columns(column_count, gap="small")
            for offset, item in enumerate(items[row_start:row_start + column_count]):
                with columns[offset]:
                    render_catalog_card(item, (page - 1) * page_size + row_start + offset)


def render_catalog_preview(item: dict[str, Any]) -> None:
    with st.container(key="catalog_preview"):
        st.button("큰 이미지 닫기 · 8열로 보기", key="close_catalog_preview", on_click=close_catalog_preview)
        render_interactive_image(
            _full_source(item), viewer_key="catalog_preview_viewer",
            alt="전체 보기에서 선택한 착장", frame_class="mf-selected-frame", height=680,
        )
        mood = html.escape(str(item.get("mood", "미분류")))
        st.markdown(f'<p class="mf-gallery-preview-mood">{mood}</p>', unsafe_allow_html=True)


def render_catalog(dataframe: pd.DataFrame) -> None:
    st.button("← 추천 화면으로", key="close_catalog", on_click=close_catalog)
    st.markdown(
        '<div class="mf-gallery-heading"><h2>전체 스타일 둘러보기</h2>'
        '<p>무드와 의류 분류로 찾아보세요. 이미지를 누르면 왼쪽에서 크게 볼 수 있습니다.</p></div>',
        unsafe_allow_html=True,
    )
    mood, category = render_catalog_filters(dataframe)
    page_size = 60
    rows, total, page = catalog_image_page(dataframe, mood, category, st.session_state.get("catalog_page", 1), page_size)
    st.session_state["catalog_page"] = page
    page_count = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size + 1 if total else 0
    end = min(page * page_size, total)
    st.caption(f"조건에 맞는 이미지 {total:,}장 · {start:,}–{end:,}장 표시 · 중복 사진 제외")
    if not total:
        st.info("선택한 조건에 맞는 이미지가 없습니다. 필터를 바꾸거나 초기화해 주세요.")
        return
    previous, page_control, following = st.columns([1, 2, 1], vertical_alignment="bottom")
    with previous:
        st.button("← 이전", key="catalog_previous", on_click=move_catalog_page, args=(-1,), disabled=page == 1, use_container_width=True)
    with page_control:
        st.number_input("페이지", min_value=1, max_value=page_count, step=1, key="catalog_page")
        st.caption(f"총 {page_count:,}페이지 · 페이지당 {page_size}장")
    with following:
        st.button("다음 →", key="catalog_next", on_click=move_catalog_page, args=(1,), disabled=page == page_count, use_container_width=True)
    selected = st.session_state.get("catalog_selected_item")
    if selected:
        with st.container(key="catalog_split"):
            preview_column, grid_column = st.columns([1, 1.15], gap="large", vertical_alignment="top")
            with preview_column:
                render_catalog_preview(selected)
            with grid_column:
                render_catalog_grid(rows, page, page_size, column_count=4)
    else:
        render_catalog_grid(rows, page, page_size, column_count=8)
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.button("← 이전 페이지", key="catalog_previous_bottom", on_click=move_catalog_page, args=(-1,), disabled=page == 1, use_container_width=True)
    with bottom_right:
        st.button("다음 페이지 →", key="catalog_next_bottom", on_click=move_catalog_page, args=(1,), disabled=page == page_count, use_container_width=True)


def main() -> None:
    dataframe, is_demo_data = load_metadata(tuple(get_model_signature(path) for path in METADATA_PATHS))
    uploaded_file = None

    st.markdown('<div class="mf-nav"><div class="mf-logo">Mood<span>Fit</span> AI</div><div class="mf-nav-note">Editorial intelligence / 2026</div></div>', unsafe_allow_html=True)

    # Keep the uploader mounted so returning from the catalog preserves the photo.
    if st.session_state.get("show_catalog", False):
        st.markdown('<style>.st-key-look_workspace {display:none !important;}</style>', unsafe_allow_html=True)
    with st.container(key="look_workspace"):
        # Shared header row: both image previews start below its tallest column.
        left, right = st.columns([1, 1], gap="large")
        preview_src = ""
        with left:
            st.markdown('<div class="mf-kicker">Your daily edit, intelligently styled</div>', unsafe_allow_html=True)
            st.markdown('<h1 class="mf-title">Wear the mood<br>you mean.</h1>', unsafe_allow_html=True)
            st.markdown('<p class="mf-lede">착장 사진을 업로드하면 오늘의 무드를 읽고, <br>당신에게 어울리는 K-Fashion을 큐레이션합니다.</p>', unsafe_allow_html=True)

            st.markdown('<div class="mf-panel-title">01 / UPLOAD YOUR LOOK</div>', unsafe_allow_html=True)

        # Align the upload control and the empty recommendation box in one row.
        upload_controls, recommendation_controls = st.columns([1, 1], gap="large")
        with upload_controls:
            with st.container(key="upload_control"):
                uploaded_file = st.file_uploader("착장 이미지 업로드", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    uploaded_file.seek(0)
                    with Image.open(uploaded_file) as uploaded_image:
                        # Re-encode a preview; a very tall source never controls layout height.
                        preview = uploaded_image.convert("RGB")
                        preview.thumbnail((2400, 2400))
                        preview_bytes = io.BytesIO()
                        preview.save(preview_bytes, format="JPEG", quality=90)
                    preview_src = "data:image/jpeg;base64," + base64.b64encode(preview_bytes.getvalue()).decode("ascii")
                except (OSError, ValueError):
                    st.error("이미지를 읽을 수 없습니다. JPG 또는 PNG 파일을 사용해 주세요.")

        upload_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest() if uploaded_file else ""
        previous_upload = st.session_state.get("upload_signature")
        if previous_upload != upload_signature:
            st.session_state["upload_signature"] = upload_signature
            st.session_state["selected_item"] = None
            st.session_state["detail_mode"] = "full"
            st.session_state.pop("upload_analysis", None)
            if previous_upload is not None:
                _clear_query_params()

        model, model_error = None, ""
        analysis, analysis_error = None, ""
        if uploaded_file:
            with st.spinner("착장의 무드를 분석하고 있어요..."):
                model, model_error = load_model_safely(get_model_signature())
                analysis, analysis_error = analyze_upload(uploaded_file, model)
        if analysis and is_demo_data and model is not None:
            dataframe = _attach_demo_moods(dataframe, model)

        # Stable across card clicks; a different upload receives a different random seed.
        recommendation_seed = int(upload_signature[:8], 16) if upload_signature else 42
        top1_items = _pick_mood_recommendations(
            dataframe, analysis["mood_scores"], count=5, random_state=recommendation_seed,
        )["similar"] if analysis and not dataframe.empty else []
        apply_query_selection(dataframe)
        selected_item = ensure_selected_recommendation(top1_items, dataframe)
        with right:
            render_analysis_panel(uploaded_file, analysis, analysis_error or model_error)
        with recommendation_controls:
            if selected_item:
                st.markdown('<div class="mf-panel-title">선택된 추천 이미지</div>', unsafe_allow_html=True)
            else:
                render_selected_recommendation(None)

        # Create an image row only when an actual preview/selection is available.
        if preview_src or selected_item:
            image_left, image_right = st.columns([1, 1], gap="large")
            with image_left:
                if preview_src:
                    render_interactive_image(
                        preview_src, viewer_key="upload_image_viewer",
                        alt="업로드한 착장", frame_class="mf-upload-frame",
                    )
            with image_right:
                if selected_item:
                    render_selected_recommendation(selected_item, show_title=False)

    mood_options = catalog_filter_options(dataframe, "mood")
    category_options = catalog_filter_options(dataframe, "category")
    diagnosis_key = (upload_signature, model.checkpoint_sha256) if analysis else None
    previous_diagnosis = st.session_state.get("diagnosis_filter_key")
    if analysis and previous_diagnosis != diagnosis_key:
        st.session_state["mood_filter"] = analysis["mood"] if analysis["mood"] in mood_options else "전체"
        st.session_state["diagnosis_filter_key"] = diagnosis_key
    elif not analysis and previous_diagnosis is not None:
        st.session_state["mood_filter"] = "전체"
        st.session_state["diagnosis_filter_key"] = None
    if st.session_state.get("show_catalog", False):
        render_catalog(dataframe)
        return

    if analysis and not dataframe.empty:
        st.markdown(
            f'<div class="mf-section"><div><h2>Top Mood Edit</h2>'
            f'<p>{html.escape(analysis["mood"])} 1순위 무드 의류 랜덤 추천</p></div>'
            '<span class="mf-pill">TOP 1 · 5 ITEMS</span></div>',
            unsafe_allow_html=True,
        )
        if top1_items:
            columns = st.columns(5, gap="medium")
            for index, item in enumerate(top1_items):
                with columns[index % 5]:
                    render_card(item, "top1", index)
        else:
            st.info("진단된 무드에 해당하는 추천 아이템이 아직 없습니다. 아래 필터에서 다른 무드를 선택해 주세요.")

    if analysis and not dataframe.empty:
        st.button("더보기 · 진단된 무드 전체", key="more_top_mood", on_click=open_catalog, args=(analysis["mood"], "전체"))

    st.divider()

    st.markdown('<div class="mf-section"><div><h2>Explore the edit</h2><p>무드와 의류 분류를 선택해 아래 추천 목록을 좁혀보세요.</p></div></div>', unsafe_allow_html=True)
    selected_mood, selected_category = render_catalog_filters(dataframe)
    st.button("더보기 · 전체 스타일 보기", key="more_catalog", on_click=open_catalog, disabled=dataframe.empty)

    if is_demo_data:
        st.info("현재는 로컬 이미지 폴더를 사용하는 데모 카탈로그입니다. AI Hub JSON 라벨과 metadata가 준비되면 정식 분류로 전환됩니다.")
    if dataframe.empty:
        st.info("추천 데이터가 아직 없습니다. 이미지와 메타데이터를 준비해 주세요.")
        return

    # No gender restriction: the current catalog has no usable gender labels.
    matching_items = filter_catalog(
        dataframe, selected_gender="전체",
        selected_category=selected_category, selected_mood=selected_mood,
    )
    match_count = matching_items["id"].nunique() if "id" in matching_items else len(matching_items)
    st.caption(f"조건에 맞는 아이템 {match_count:,}개 · 무드: {selected_mood} · 의류 분류: {selected_category}")
    if matching_items.empty:
        st.info("선택한 조건에 맞는 아이템이 없습니다. 다른 무드·의류 분류를 선택하거나 필터를 초기화해 주세요.")
        return

    user_embedding = analysis["embedding"] if analysis else None
    item_embeddings, _ = load_item_embeddings(dataframe, model)
    sections = get_recommendation_sections(
        user_embedding, selected_category, dataframe,
        item_embeddings=item_embeddings, selected_gender="전체",
        selected_mood=selected_mood, random_state=recommendation_seed,
    )
    has_similarity = bool(sections["similar"]) and all("similarity" in item for item in sections["similar"])
    section_title = "AI Similarity" if has_similarity else "Mood Picks" if analysis else "Curated Picks"
    section_description = "선택한 조건 안에서 입력한 착장과 가까운 아이템" if has_similarity else "선택한 무드·의류 분류에 맞는 아이템 · 유사도 순위가 아닙니다"
    st.markdown(f'<div class="mf-section"><div><h2>{section_title}</h2><p>{section_description}</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
    similar_items = sections["similar"]
    if similar_items:
        columns = st.columns(5, gap="medium")
        for index, item in enumerate(similar_items):
            with columns[index % 5]:
                render_card(item, "similar", index)
    else:
        st.markdown('<div class="mf-empty">현재 조건에 맞는 추천 아이템이 없습니다.</div>', unsafe_allow_html=True)

    st.button("더보기 · 같은 조건의 전체 이미지", key="more_similar", on_click=open_catalog)

    st.markdown('<div class="mf-section"><div><h2>Trending Discovery</h2><p>같은 무드·의류 분류 조건에서 새롭게 발견한 아이템</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
    discovery_items = sections["discovery"]
    if discovery_items:
        columns = st.columns(5, gap="medium")
        for index, item in enumerate(discovery_items):
            with columns[index % 5]:
                render_card(item, "discovery", index)
    else:
        st.markdown('<div class="mf-empty">디스커버리 아이템이 없습니다.</div>', unsafe_allow_html=True)

    st.button("더보기 · 스타일 더 찾아보기", key="more_discovery", on_click=open_catalog)


if __name__ == "__main__":
    main()
