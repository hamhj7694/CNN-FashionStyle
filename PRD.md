# MoodFit AI — 패션 큐레이션 웹앱 MVP PRD

## 1. 개요

MoodFit AI는 사용자가 업로드한 착장 사진의 스타일 무드를 분석하고, AI Hub K-Fashion 메타데이터와 전이학습 모델을 이용해 개인화된 의상을 큐레이션하는 Streamlit MVP다.

- 목표: 업로드 → 무드 진단 → 성별/카테고리 필터 → 유사 추천 및 트렌디 디스커버리
- 디자인: StyleRoom을 오마주한 모던 SaaS 에디토리얼 UI
- 컬러: 차콜 `#0F172A`, 오프화이트 `#F8FAFC`, 포인트 라벤더/웜 그레이
- UI: Streamlit 기본 위젯을 기반으로 `streamlit-shadcn-ui` 카드·뱃지를 선택적으로 사용
- 핵심 원칙: 원본 이미지는 보존하고, 화면에 필요한 추천 아이템만 최초 렌더링 시 크롭·캐싱

## 2. 사용자 흐름

1. 사용자가 JPG/PNG 착장 이미지를 업로드한다.
2. 모델 인터페이스가 준비되면 무드 분석 결과와 사용자 이미지 임베딩을 생성한다. 현재 MVP는 모델 스텁 및 안전한 데모 진단을 제공한다.
3. 성별을 `전체`, `여성 (공용 포함)`, `남성 (공용 포함)` 중에서 선택한다.
4. 카테고리와 무드를 추가로 선택한다.
5. 사용자 이미지 임베딩과 후보 이미지 임베딩을 비교해 유사도 순 TOP 5와 트렌디 디스커버리 TOP 5를 확인한다.
6. 카드가 실제로 렌더링될 때만 `recommend_pool/processed_crops/crop_{id}.jpg`를 생성한다.

## 3. 디렉토리 계약

```text
MoodFit-AI/
├── recommend_pool/
│   ├── raw_data/
│   │   ├── images/       # AI Hub 원본 이미지
│   │   └── labels/       # AI Hub JSON 라벨
│   ├── processed_crops/  # Lazy Cropping 캐시
│   ├── metadata.csv      # JSON annotation 단위 상세 카탈로그
│   ├── gallery_index.csv  # 전신 이미지 단위 임베딩 매핑 카탈로그
│   └── item_embeddings_web.npy  # gallery_index.csv와 동일한 순서의 벡터
├── app.py
├── model.py
├── preprocess.py
├── curation.py
├── requirements.txt
└── PRD.md
```

`preprocess.py`는 현재 작업 폴더의 레거시 데이터도 사용할 수 있도록 명시적 경로 인자를 지원한다. CSV의 `bbox`는 JSON 문자열 `[x1, y1, x2, y2]`로 저장한다.

## 4. 기능 요구사항

### 전처리

- `recommend_pool/raw_data/labels/**/*.json`을 재귀적으로 읽는다.
- 이미지 경로, 성별, 카테고리, 무드를 정규화한다.
- 다음 컬럼을 가진 `recommend_pool/metadata.csv`를 생성한다.

  `id`, `image_filename`, `original_img_path`, `original_json_path`, `gender`, `category`, `category_detail`, `mood`, `bbox`

- JSON annotation 단위로 행을 생성한다. 하나의 이미지에 여러 의류 부위가 있으면 각 bbox를 별도 행으로 저장한다.
- `recommend_pool/gallery_index.csv`는 전신 이미지 단위로 중복을 제거해 생성한다. 유사 추천은 이 카탈로그를 기준으로 하므로 같은 전신 이미지가 TOP 5에 중복 노출되지 않는다.
- 전처리 단계에서는 이미지를 크롭하지 않는다.
- 누락·손상된 JSON은 전체 파이프라인을 중단하지 않고 건너뛴다.

### 추천 및 캐싱

- 여성 선택 시 `여성`, `공용`만 허용한다.
- 남성 선택 시 `남성`, `공용`만 허용한다.
- 유사 추천은 모델 임베딩과 cosine similarity를 사용하고, 임베딩이 없으면 deterministic fallback을 사용한다.
- Lazy Cropping은 캐시 파일이 있으면 즉시 반환하고, 없을 때만 원본에서 크롭·512×512 리사이즈·저장한다.

### UI

- 상단 브랜드 내비게이션과 업로드/분석 2분할 캔버스를 제공한다.
- 진단 무드, 추천 성별, 카테고리를 분석 패널에 표시한다.
- 성별·카테고리·무드 필터를 제공한다.
- AI 유사 추천 TOP 5와 트렌디 디스커버리 TOP 5를 제공한다.
- `streamlit-shadcn-ui`가 설치된 환경에서는 `ui.card`, `ui.badge`, `ui.button`을 사용하고, 미설치 환경에서는 동일한 CSS 카드로 폴백한다.

## 5. 모델 결과와 이미지 매핑

모델의 분류 결과만으로는 유사 이미지를 찾을 수 없다. 예를 들어 `밀리터리 87%`와 같은 무드 확률은 진단·필터 표시용이고, 유사 추천에는 이미지 특징 벡터(embedding)가 필요하다.

### 오프라인 후보 인덱스

모든 추천 후보 전신 이미지에 동일한 feature extractor를 적용하고, 결과를 `item_embeddings_web.npy`에 저장한다. 벡터의 행 순서는 반드시 `gallery_index.csv`와 동일해야 한다.

```text
gallery_index.csv 행 0  ↔  item_embeddings_web.npy 행 0  ↔  3725.jpg
gallery_index.csv 행 1  ↔  item_embeddings_web.npy 행 1  ↔  12821.jpg
```

`gallery_index.csv`의 최소 컬럼은 다음과 같다.

`embedding_index`, `id`, `original_img_path`, `original_json_path`

### 온라인 유사도 검색

사용자가 업로드한 이미지에서 query embedding을 만든 뒤 후보 벡터 전체와 cosine similarity를 계산한다.

```python
query_embedding = extract_image_embedding(uploaded_image, model)
scores = cosine_similarity([query_embedding], item_embeddings)[0]
top_indices = np.argsort(scores)[::-1][:5]
recommended = gallery_index.iloc[top_indices].copy()
recommended["similarity"] = scores[top_indices]
```

모델이 이미지 파일을 반환하는 것이 아니라 다음 순서로 연결한다.

```text
모델 → query embedding
유사도 계산 → top_indices
top_indices → gallery_index.csv의 이미지 경로
이미지 경로 → 연결된 JSON
JSON bbox → 상세 부위 Lazy Crop
```

유사 추천 결과에는 반드시 `id`, `original_img_path`, `original_json_path`, `similarity`를 포함한다. 상세 화면에서 해당 전신 이미지의 의류 부위를 보여줄 때는 `id` 또는 `original_img_path`로 `metadata.csv`의 관련 annotation 행을 조회하고 bbox를 사용한다.

## 6. 모델 연동 계약

모델 학습은 사용자의 `CNN_model_PyTorch.ipynb`에서 수행하고, `model.py`는 저장된 가중치를 웹앱에 연결한다. 모델 구조와 전처리는 학습 노트북과 동일하게 유지한다.

```python
def load_feature_extractor():
    """Model/fashion_mood_model.pth를 읽어 eval 상태의 PyTorch 모델을 반환한다."""

def extract_image_embedding(image, model):
    """PIL 이미지 또는 배열을 임베딩 벡터로 변환한다."""

def predict_mood(image, model):
    """mood, confidence, mood_scores를 반환한다."""

def analyze_image(image, model):
    """한 번의 추론으로 무드 결과와 추천용 embedding을 함께 반환한다."""
```

`extract_image_embedding`은 마지막 분류층 직전의 256차원 벡터를 L2 정규화해 반환한다. 무드 확률과 검색 벡터는 구분한다. 기본 클래스 순서는 학습 노트북의 24개 `ImageFolder.classes` 순서와 같다. 전처리는 RGB → 224×224 Resize → ToTensor → ImageNet Normalize다.

웹앱은 실제 예측 무드·점수를 표시하고, 성별·의류 종류는 모델이 예측한 것처럼 표시하지 않는다. 모델이 없거나 저장 중이면 안내를 표시한다. 모델 파일 생성·변경 시 캐시 키가 바뀌며 같은 이미지와 모델에 대한 분석은 세션 안에서 재사용한다.

유사 추천 전에 앱에서 사용하는 웹 카탈로그 CSV로 `python model.py --build-index recommend_pool/metadata.csv`를 실행한다. `metadata_web.csv` 또는 `metadata_r2.csv`를 사용하는 앱이라면 해당 파일을 인자로 전달한다. 생성된 `item_embeddings_web.npy`와 `.json`에는 모델 체크섬·전처리 버전·카탈로그 행 순서·벡터 체크섬을 함께 기록한다. 검증되지 않은 벡터를 사용하거나 무작위 결과를 유사도 순위로 표시하지 않는다. 실행 예시는 `MODEL_INTEGRATION.md`에 정리한다.

향후 모델 분석 결과는 다음처럼 확장할 수 있다.

```python
{
    "mood": "밀리터리",
    "mood_scores": {...},
    "embedding": [...],
}
```

## 7. MVP 완료 기준

- `python preprocess.py` 또는 명시적 경로 인자로 metadata CSV를 생성할 수 있다.
- `streamlit run app.py`가 metadata CSV가 없어도 안내 화면 또는 로컬 이미지 데모로 실행된다.
- JSON 라벨과 이미지가 있는 경우 원본을 미리 복제하지 않고 추천 시점에만 캐시가 생성된다.
- `gallery_index.csv`와 `item_embeddings_web.npy`의 행 매핑으로 유사 전신 이미지 TOP 5를 조회할 수 있다.
- TOP 5 결과에서 전신 이미지, 연결 JSON, bbox 크롭 상세보기를 순서대로 확인할 수 있다.
- 모델 스텁을 실제 구현으로 교체해도 앱·큐레이션 모듈의 공개 인터페이스를 바꾸지 않는다.

## 8. 이미지 스토리지 연동

GitHub 저장소에는 코드와 메타데이터 구조만 보관하고, 크롭 이미지는 Cloudflare R2에 저장한다.

- `r2_storage.py`: Cloudflare R2(S3-compatible) 클라이언트와 업로드 유틸리티
- `upload_crops_to_r2.py`: 전신 썸네일을 `full/`, 아이템 크롭을 `crops/` prefix로 업로드하고 `metadata_r2.csv` 생성
- `metadata_r2.csv`: 기존 메타데이터에 `full_image_url`과 `image_url`을 추가한 배포용 카탈로그
- 전신 썸네일은 원본 비율을 유지한 웹 표시용 축소본이며 학습용 원본과 분리한다.
- R2 Access Key와 Secret은 환경변수로만 주입하며 GitHub에 저장하지 않는다.
- 앱은 `image_url`을 우선 사용하고, URL이 없는 로컬 개발 환경에서는 기존 `processed_crops`를 사용한다.

## 9. 카탈로그 분류 계층

웹 카탈로그의 탐색 순서는 다음과 같이 고정한다.

```text
무드
└── 성별 범위
    ├── 여성 (여성 + 공용)
    │   └── 의류 분류
    └── 남성 (남성 + 공용)
        └── 의류 분류
```

저장 데이터의 성별 값은 `여성`, `남성`, `공용` 3개를 유지한다. UI에서는 공용 아이템을 여성·남성 두 범위에 포함시켜 사용한다. 의류 분류의 상위·하위 카테고리는 AI Hub JSON 라벨을 집계한 뒤 확정하며, 임의의 카테고리명을 먼저 하드코딩하지 않는다.
