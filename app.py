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
    .mf-image-frame { position:relative; overflow:hidden; border-radius:16px; background:#E2E8F0; aspect-ratio:4/5; }
    .mf-image-frame img { display:block; width:100%; height:100%; object-fit:cover; transition:transform .28s ease; }
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
    return dataframe[required].fillna("")


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
    st.session_state["selected_item"] = item
    st.session_state["detail_mode"] = "full"
    _rerun()


def render_card(item: dict[str, Any], section_key: str, index: int) -> None:
    full_source = _full_source(item)
    select_query = f"select_id={quote(str(item.get('id', '')), safe='')}"
    render_image_hover(full_source, f"{item.get('category', 'item')} full look", query=select_query)

    category = html.escape(str(item.get("category", "기타")))
    mood = html.escape(str(item.get("mood", "미분류")))
    score = item.get("similarity")
    score_text = f"유사도 {float(score) * 100:.0f}%" if score is not None else "curated pick"
    if HAS_SHADCN and hasattr(ui, "card"):
        try:
            ui.card(
                title=f"{category} · {mood}",
                content=f"{score_text} · MoodFit edit",
                key=f"card_{section_key}_{index}",
            )
        except Exception:
            pass
    else:
        st.markdown(
            f'<div class="mf-card-copy"><div class="mf-card-meta"><span>{category}</span><span>{score_text}</span></div><div class="mf-card-title">{mood} edit</div></div>',
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
    uploaded_file: Any, selected_item: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None, analysis_error: str = "",
) -> None:
    if selected_item:
        header, close = st.columns([.82, .18])
        with header:
            st.markdown('<div class="mf-panel-title">ITEM DETAIL / SELECTED EDIT</div>', unsafe_allow_html=True)
        with close:
            if st.button("×", key="close_detail", use_container_width=True):
                st.session_state["selected_item"] = None
                st.session_state["detail_mode"] = "full"
                _clear_query_params()
                _rerun()

        item = selected_item
        full_source = _full_source(item)
        crop_source = _crop_source(item)
        detail_mode = st.session_state.get("detail_mode", "full")
        main_source = crop_source if detail_mode == "crop" and crop_source else full_source
        render_image_hover(main_source, "selected item detail")
        st.markdown(
            f'<div class="mf-analysis"><span class="mf-pill">{html.escape(str(item.get("mood", "미분류")))}</span><h3>{html.escape(str(item.get("category", "추천 아이템")))}</h3><p>선택한 추천 아이템의 전신 이미지와 아이템 크롭을 확인할 수 있습니다.</p><div class="mf-stat"><span>성별 범위</span><b>{html.escape(str(item.get("gender", "공용")))}</b></div><div class="mf-stat"><span>아이템 ID</span><b>{html.escape(str(item.get("id", "-")))}</b></div></div>',
            unsafe_allow_html=True,
        )
        if crop_source:
            st.markdown('<div class="mf-panel-title">CROP PREVIEW / HOVER TO INSPECT</div>', unsafe_allow_html=True)
            crop_query = f"select_id={quote(str(item.get('id', '')), safe='')}&view=crop"
            render_image_hover(crop_source, "item crop", query=crop_query)
            if detail_mode == "crop" and st.button("전신 이미지로 돌아가기", key=f"full_detail_{item.get('id')}", use_container_width=True):
                st.session_state["detail_mode"] = "full"
                _clear_query_params()
                _rerun()
        return

    st.markdown('<div class="mf-panel-title">02 / MOOD DIAGNOSIS</div>', unsafe_allow_html=True)
    if uploaded_file and analysis:
        render_badge("MOOD DETECTED", "mood_detected", dark=True)
        mood = html.escape(analysis["mood"])
        alternatives = sorted(analysis["mood_scores"].items(), key=lambda item: item[1], reverse=True)[1:3]
        secondary = html.escape(" · ".join(f"{name} {score:.1%}" for name, score in alternatives))
        st.markdown(
            f'<div class="mf-analysis"><h3>{mood}</h3>'
            '<p>업로드한 착장에서 감지한 패션 무드입니다. 성별과 의류 분류는 아래 필터에서 선택해 보세요.</p>'
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
        st.markdown('<div class="mf-analysis"><div style="font-size:2.5rem;">✦</div><h3>Start with a look</h3><p>이미지를 업로드하면 무드 진단과 개인화된 아이템 편집이 이곳에 나타납니다.</p></div>', unsafe_allow_html=True)


def main() -> None:
    dataframe, is_demo_data = load_metadata(tuple(get_model_signature(path) for path in METADATA_PATHS))
    uploaded_file = None

    st.markdown('<div class="mf-nav"><div class="mf-logo">Mood<span>Fit</span> AI</div><div class="mf-nav-note">Editorial intelligence / 2026</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="mf-kicker">Your daily edit, intelligently styled</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="mf-title">Wear the mood<br>you mean.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mf-lede">착장 사진을 업로드하면 오늘의 무드를 읽고, 당신에게 어울리는 K-Fashion 아이템을 큐레이션합니다.</p>', unsafe_allow_html=True)

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="mf-panel-title">01 / UPLOAD YOUR LOOK</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("착장 이미지 업로드", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file:
            try:
                uploaded_file.seek(0)
                with Image.open(uploaded_file) as uploaded_image:
                    st.image(uploaded_image.convert("RGB"), use_container_width=True, caption="Uploaded look")
            except (OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. JPG 또는 PNG 파일을 사용해 주세요.")
        else:
            st.markdown('<div class="mf-preview-empty" style="margin-top:1rem;"><strong>착장 미리보기</strong><small>상단 Upload 영역에서 이미지를 선택하면<br>이곳에 미리보기가 표시됩니다.</small></div>', unsafe_allow_html=True)

    upload_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest() if uploaded_file else ""
    previous_upload = st.session_state.get("upload_signature")
    upload_changed = previous_upload != upload_signature
    if upload_changed:
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

    apply_query_selection(dataframe)
    selected_item = st.session_state.get("selected_item")
    with right:
        render_analysis_panel(uploaded_file, selected_item, analysis, analysis_error or model_error)

    st.markdown('<div class="mf-section"><div><h2>Explore the edit</h2><p>무드에서 시작해 성별 범위와 의류 분류로 큐레이션을 좁혀보세요.</p></div></div>', unsafe_allow_html=True)
    filter_a, filter_b, filter_c = st.columns([1, 1.25, 1])
    with filter_a:
        mood_options = ["전체"] + sorted([str(value) for value in dataframe["mood"].unique() if str(value)]) if not dataframe.empty else ["전체"]
        # 새 업로드/첫 분석 시 예측 무드를 기본값으로 쓰되 이후 사용자의 변경은 유지한다.
        diagnosis_key = (upload_signature, model.checkpoint_sha256) if analysis else None
        if analysis and st.session_state.get("diagnosis_filter_key") != diagnosis_key:
            st.session_state["mood_filter"] = analysis["mood"] if analysis["mood"] in mood_options else "전체"
            st.session_state["diagnosis_filter_key"] = diagnosis_key
        if st.session_state.get("mood_filter", "전체") not in mood_options:
            st.session_state["mood_filter"] = "전체"
        selected_mood = st.selectbox("무드", mood_options, key="mood_filter", label_visibility="collapsed")
    with filter_b:
        selected_gender = st.radio("성별", ["전체", "여성 (공용 포함)", "남성 (공용 포함)"], horizontal=True, label_visibility="collapsed")
    with filter_c:
        category_options = ["전체"] + sorted([str(value) for value in dataframe["category"].unique() if str(value)]) if not dataframe.empty else ["전체"]
        selected_category = st.selectbox("의류 분류", category_options, label_visibility="collapsed")

    if is_demo_data:
        st.info("현재는 로컬 이미지 폴더를 사용하는 데모 카탈로그입니다. AI Hub JSON 라벨과 metadata가 준비되면 정식 분류로 전환됩니다.")
    if dataframe.empty:
        st.markdown('<div class="mf-empty">추천 데이터가 아직 없습니다. recommend_pool/raw_data/images와 labels를 준비한 뒤 preprocess.py를 실행해 주세요.</div>', unsafe_allow_html=True)
        return

    user_embedding = analysis["embedding"] if analysis else None
    item_embeddings, index_error = load_item_embeddings(dataframe, model)
    if analysis and index_error:
        st.info(f"{index_error} 현재는 선택한 필터의 아이템을 둘러볼 수 있습니다.")
    sections = get_recommendation_sections(
        user_embedding,
        selected_category,
        dataframe,
        item_embeddings=item_embeddings,
        selected_gender=selected_gender,
        selected_mood=selected_mood,
        random_state=42,
    )

    has_similarity = user_embedding is not None and item_embeddings is not None
    section_title = "AI Similarity" if has_similarity else "Curated Picks"
    section_description = "입력한 전신 이미지와 가장 가까운 전신 추천" if has_similarity else "선택한 필터에서 발견한 아이템 · 유사도 순위가 아닙니다"
    st.markdown(f'<div class="mf-section"><div><h2>{section_title}</h2><p>{section_description}</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
    similar_items = sections["similar"]
    if similar_items:
        columns = st.columns(5, gap="medium")
        for index, item in enumerate(similar_items):
            with columns[index % 5]:
                render_card(item, "similar", index)
    else:
        st.markdown('<div class="mf-empty">현재 필터와 맞는 유사 아이템이 없습니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="mf-section"><div><h2>Trending Discovery</h2><p>같은 필터 안에서 새롭게 발견한 랜덤 아이템</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
    discovery_items = sections["discovery"]
    if discovery_items:
        columns = st.columns(5, gap="medium")
        for index, item in enumerate(discovery_items):
            with columns[index % 5]:
                render_card(item, "discovery", index)
    else:
        st.markdown('<div class="mf-empty">디스커버리 아이템이 없습니다.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
