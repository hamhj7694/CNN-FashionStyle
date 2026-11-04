"""MoodFit AI Streamlit MVP."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from curation import get_recommendation_sections, lazy_crop_item

try:
    import streamlit_shadcn_ui as ui

    HAS_SHADCN = True
except ImportError:
    ui = None
    HAS_SHADCN = False


ROOT = Path(__file__).resolve().parent
METADATA_PATHS = [ROOT / "recommend_pool" / "metadata.csv", ROOT / "metadata.csv"]
CROP_DIR = ROOT / "recommend_pool" / "processed_crops"

st.set_page_config(page_title="MoodFit AI", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink: #0F172A; --paper: #F8FAFC; --muted: #64748B; --line: #E2E8F0; --accent: #A78BFA; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid='stHeader'] { background: rgba(248,250,252,.86); }
    .block-container { max-width: 1440px; padding: 1.5rem 3rem 4rem; }
    .mf-nav { display:flex; align-items:center; justify-content:space-between; padding: .6rem 0 1.4rem; border-bottom: 1px solid var(--line); margin-bottom: 2.4rem; }
    .mf-logo { font-family: 'Playfair Display', Georgia, serif; font-size: 1.65rem; letter-spacing: -.03em; }
    .mf-logo span { color: var(--accent); }
    .mf-nav-note { color: var(--muted); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
    .mf-kicker { color: #7C3AED; font-size: .74rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .6rem; }
    .mf-title { font-family: 'Playfair Display', Georgia, serif; font-size: clamp(2.2rem, 4vw, 4.3rem); line-height: 1.02; letter-spacing: -.04em; margin: 0; }
    .mf-lede { color: var(--muted); max-width: 560px; font-size: 1rem; line-height: 1.7; margin-top: 1rem; }
    .mf-panel { background: white; border: 1px solid var(--line); border-radius: 22px; padding: 1.25rem; min-height: 365px; box-shadow: 0 18px 50px rgba(15,23,42,.04); }
    .mf-panel-title { font-size: .76rem; font-weight: 700; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-bottom: .9rem; }
    .mf-analysis { height: 100%; display:flex; flex-direction:column; justify-content:center; padding: 1rem; }
    .mf-analysis h3 { font-family:'Playfair Display', Georgia, serif; font-size: 2rem; margin: .9rem 0 .4rem; }
    .mf-analysis p { color: var(--muted); line-height: 1.6; margin-bottom: 1.5rem; }
    .mf-stat { display:flex; justify-content:space-between; border-top: 1px solid var(--line); padding: .8rem 0; font-size: .9rem; }
    .mf-stat span:first-child { color: var(--muted); }
    .mf-pill { display:inline-block; background:#EDE9FE; color:#6D28D9; border-radius:999px; padding:.38rem .72rem; font-size:.74rem; font-weight:700; }
    .mf-pill-dark { background:var(--ink); color:white; }
    .mf-section { display:flex; align-items:end; justify-content:space-between; margin: 3.2rem 0 1rem; }
    .mf-section h2 { font-family:'Playfair Display', Georgia, serif; font-size: 2rem; margin:0; letter-spacing:-.03em; }
    .mf-section p { color:var(--muted); margin:.35rem 0 0; font-size:.9rem; }
    .mf-card-copy { padding: .8rem .1rem .2rem; }
    .mf-card-meta { color:var(--muted); font-size:.72rem; display:flex; justify-content:space-between; gap:.3rem; }
    .mf-card-title { font-weight:700; margin:.35rem 0 .2rem; font-size:.92rem; }
    .mf-empty { border: 1px dashed #CBD5E1; border-radius: 18px; padding: 2rem; color: var(--muted); text-align:center; background:white; }
    .mf-preview-empty { min-height: 260px; border: 1px solid var(--line); border-radius: 16px; padding: 2rem; color: var(--muted); text-align:center; background: #FFFFFF; display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .mf-preview-empty strong { color: var(--ink); font-size: 1rem; margin-bottom: .35rem; }
    .mf-preview-empty small { color: #94A3B8; font-size: .78rem; }
    div[data-testid='stFileUploaderDropzone'] { border: 1px dashed #94A3B8; background:#F8FAFC; border-radius: 16px; min-height: 260px; }
    div[data-testid='stFileUploaderDropzone']:hover { border-color:#8B5CF6; background:#F5F3FF; }
    div[data-testid='stRadio'] > div { gap:.5rem; }
    div[data-testid='stRadio'] label { background:white; border:1px solid var(--line); border-radius:999px; padding:.32rem .8rem; }
    .stButton > button { border-radius:999px; border:1px solid var(--line); background:white; color:var(--ink); }
    .stButton > button:hover { border-color:#8B5CF6; color:#6D28D9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metadata() -> tuple[pd.DataFrame, bool]:
    for path in METADATA_PATHS:
        if path.is_file():
            dataframe = pd.read_csv(path)
            return _normalize_metadata(dataframe), False

    # 현재 저장소에 포함된 레거시 이미지 폴더를 별도 CSV 생성 없이 데모 카탈로그로 사용한다.
    legacy_root = ROOT / "K-Fashion 이미지"
    if legacy_root.is_dir():
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
                }
            )
            if len(rows) >= 60:
                break
        if rows:
            return pd.DataFrame(rows), True
    return _normalize_metadata(pd.DataFrame()), False


def _normalize_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = ["id", "original_img_path", "original_json_path", "gender", "category", "mood", "bbox"]
    dataframe = dataframe.copy()
    legacy_columns = {"image_path": "original_img_path"}
    dataframe = dataframe.rename(columns=legacy_columns)
    for column in required:
        if column not in dataframe.columns:
            dataframe[column] = ""
    return dataframe[required].fillna("")


def render_badge(label: str, key: str, dark: bool = False) -> None:
    if HAS_SHADCN and hasattr(ui, "badge"):
        try:
            ui.badge(label, key=key)
            return
        except Exception:
            pass
    class_name = "mf-pill mf-pill-dark" if dark else "mf-pill"
    st.markdown(f'<span class="{class_name}">{html.escape(label)}</span>', unsafe_allow_html=True)


def render_card(item: dict[str, Any], section_key: str, index: int) -> None:
    crop_path = lazy_crop_item(item, CROP_DIR)
    if crop_path:
        st.image(crop_path, use_container_width=True)
    else:
        st.markdown('<div class="mf-empty">이미지 준비 중</div>', unsafe_allow_html=True)

    category = html.escape(str(item.get("category", "기타")))
    mood = html.escape(str(item.get("mood", "미분류")))
    score = item.get("similarity")
    score_text = f"유사도 {float(score) * 100:.0f}%" if score is not None else "curated pick"
    if HAS_SHADCN and hasattr(ui, "card"):
        try:
            ui.card(
                title=f"{category} · {mood}",
                content=f"{score_text}  ·  MoodFit edit",
                key=f"card_{section_key}_{index}",
            )
        except Exception:
            st.markdown(
                f'<div class="mf-card-copy"><div class="mf-card-meta"><span>{category}</span><span>{score_text}</span></div><div class="mf-card-title">{mood} edit</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="mf-card-copy"><div class="mf-card-meta"><span>{category}</span><span>{score_text}</span></div><div class="mf-card-title">{mood} edit</div></div>',
            unsafe_allow_html=True,
        )
    if HAS_SHADCN and hasattr(ui, "button"):
        try:
            ui.button("저장", key=f"save_{section_key}_{index}")
            return
        except Exception:
            pass
    st.button("저장", key=f"save_{section_key}_{index}")


def main() -> None:
    dataframe, is_demo_data = load_metadata()
    st.markdown(
        '<div class="mf-nav"><div class="mf-logo">Mood<span>Fit</span> AI</div><div class="mf-nav-note">Editorial intelligence / 2026</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mf-kicker">Your daily edit, intelligently styled</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="mf-title">Wear the mood<br>you mean.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mf-lede">착장 사진을 업로드하면 오늘의 무드를 읽고, 당신에게 어울리는 K-Fashion 아이템을 큐레이션합니다.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="mf-panel-title">01 / Upload your look</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("착장 이미지 업로드", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file:
            try:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(uploaded_image, use_container_width=True, caption="Uploaded look")
            except (OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. JPG 또는 PNG 파일을 사용해 주세요.")
        else:
            st.markdown('<div class="mf-preview-empty" style="margin-top:1rem;"><strong>착장 미리보기</strong><small>상단 Upload 영역에서 이미지를 선택하면<br>이곳에 미리보기가 표시됩니다.</small></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="mf-panel-title">02 / Mood diagnosis</div>', unsafe_allow_html=True)
        if uploaded_file:
            render_badge("MOOD DETECTED", "mood_detected", dark=True)
            st.markdown('<div class="mf-analysis"><h3>Modern Minimal</h3><p>정제된 실루엣과 차분한 톤이 만드는 모던 미니멀 무드입니다. 구조적인 아우터와 뉴트럴 아이템을 매칭해 보세요.</p><div class="mf-stat"><span>추천 성별</span><b>공용</b></div><div class="mf-stat"><span>핵심 카테고리</span><b>아우터</b></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mf-analysis"><div style="font-size:2.5rem;">✦</div><h3>Start with a look</h3><p>이미지를 업로드하면 무드 진단과 개인화된 아이템 편집이 이곳에 나타납니다.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="mf-section"><div><h2>Explore the edit</h2><p>성별과 무드에 맞춰 당신의 큐레이션을 좁혀보세요.</p></div></div>', unsafe_allow_html=True)
    filter_a, filter_b, filter_c = st.columns([1.25, 1, 1])
    with filter_a:
        selected_gender = st.radio("성별", ["전체", "여성 (공용 포함)", "남성 (공용 포함)"], horizontal=True, label_visibility="collapsed")
    with filter_b:
        category_options = ["전체"] + sorted([str(value) for value in dataframe["category"].dropna().unique() if str(value)]) if not dataframe.empty else ["전체"]
        selected_category = st.selectbox("카테고리", category_options, label_visibility="collapsed")
    with filter_c:
        mood_options = ["전체"] + sorted([str(value) for value in dataframe["mood"].dropna().unique() if str(value)]) if not dataframe.empty else ["전체"]
        selected_mood = st.selectbox("무드", mood_options, label_visibility="collapsed")

    if is_demo_data:
        st.info("현재는 `K-Fashion 이미지` 폴더를 사용하는 데모 카탈로그입니다. AI Hub JSON 라벨을 넣고 `python preprocess.py`를 실행하면 정식 metadata로 전환됩니다.")
    if dataframe.empty:
        st.markdown('<div class="mf-empty">추천 데이터가 아직 없습니다. `recommend_pool/raw_data/images`와 `labels`를 준비한 뒤 <code>python preprocess.py</code>를 실행해 주세요.</div>', unsafe_allow_html=True)
        return

    sections = get_recommendation_sections(
        None if not uploaded_file else [1.0],
        selected_category,
        dataframe,
        selected_gender=selected_gender,
        selected_mood=selected_mood,
        random_state=42,
    )
    if uploaded_file or not dataframe.empty:
        st.markdown('<div class="mf-section"><div><h2>AI Similarity</h2><p>당신의 무드와 가장 가까운 아이템</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
        similar_items = sections["similar"]
        if similar_items:
            columns = st.columns(5, gap="medium")
            for index, item in enumerate(similar_items):
                with columns[index % 5]:
                    render_card(item, "similar", index)
        else:
            st.markdown('<div class="mf-empty">현재 필터와 맞는 유사 아이템이 없습니다.</div>', unsafe_allow_html=True)

        st.markdown('<div class="mf-section"><div><h2>Trending Discovery</h2><p>예상 밖의 조합으로 넓혀보는 오늘의 발견</p></div><span class="mf-pill">TOP 5</span></div>', unsafe_allow_html=True)
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
