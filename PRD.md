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
2. 모델 인터페이스가 준비되면 무드·카테고리·임베딩을 반환한다. 현재 MVP는 모델 스텁 및 안전한 데모 진단을 제공한다.
3. 성별을 `전체`, `여성 (공용 포함)`, `남성 (공용 포함)` 중에서 선택한다.
4. 카테고리와 무드를 추가로 선택한다.
5. 유사 추천 TOP 5와 트렌디 디스커버리 TOP 5를 5열 카드 그리드로 확인한다.
6. 카드가 실제로 렌더링될 때만 `recommend_pool/processed_crops/crop_{id}.jpg`를 생성한다.

## 3. 디렉토리 계약

```text
MoodFit-AI/
├── recommend_pool/
│   ├── raw_data/
│   │   ├── images/       # AI Hub 원본 이미지
│   │   └── labels/       # AI Hub JSON 라벨
│   ├── processed_crops/  # Lazy Cropping 캐시
│   └── metadata.csv
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

  `id`, `original_img_path`, `original_json_path`, `gender`, `category`, `mood`, `bbox`

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

## 5. 모델 연동 계약

`model.py`는 사용자 구현 영역이다. 다음 함수 시그니처를 유지한다.

```python
def load_feature_extractor():
    """TensorFlow/Keras feature extractor를 반환한다."""

def extract_image_embedding(image, model):
    """PIL 이미지 또는 배열을 임베딩 벡터로 변환한다."""
```

향후 모델이 반환할 수 있는 추가 정보는 `mood`, `gender`, `category`, `embedding` 키를 가진 분석 결과 객체로 확장한다.

## 6. MVP 완료 기준

- `python preprocess.py` 또는 명시적 경로 인자로 metadata CSV를 생성할 수 있다.
- `streamlit run app.py`가 metadata CSV가 없어도 안내 화면 또는 로컬 이미지 데모로 실행된다.
- JSON 라벨과 이미지가 있는 경우 원본을 미리 복제하지 않고 추천 시점에만 캐시가 생성된다.
- 모델 스텁을 실제 구현으로 교체해도 앱·큐레이션 모듈의 공개 인터페이스를 바꾸지 않는다.
