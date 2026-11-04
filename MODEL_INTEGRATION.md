# 저장한 PyTorch 모델을 웹앱에서 사용하기

모델 학습은 기존 `CNN_model_PyTorch.ipynb`에서 진행한다. 노트북과
`mood_model_가져오기.py`는 이번 연동 과정에서 수정하지 않았다.

## 1. 학습이 끝나면 모델 저장 셀 실행

노트북에 작성된 다음 저장 코드를 학습 완료 후 실행한다.

```python
os.makedirs("Model", exist_ok=True)
torch.save(mood_model.state_dict(), "fashion_mood_model_pytorch.pth")
```

최종 위치는 프로젝트 폴더의 `Model/fashion_mood_model_pytorch.pth`다. 다른 디렉터리에서
노트북을 실행했다면 실제 저장된 위치를 확인해 이 경로로 옮긴다. `.pth`는 대용량
가중치라 `.gitignore`에 포함했다. 다른 PC나 배포 서버에는 별도로 전달해야 한다.

`model.py`는 ConvNeXt-Tiny와 학습 시 사용한 분류기를 다시 만든 뒤 가중치를
읽는다. 클래스 순서는 현재 노트북의 24개 클래스 순서이며, 전처리도 현재
노트북과 같은 224×224 리사이즈 및 ImageNet 정규화다. 향후 클래스 순서를
바꾼다면 모델과 함께 순서를 저장해야 한다. 다음 checkpoint 형식도 지원한다.

```python
torch.save({
    "state_dict": mood_model.state_dict(),
    "class_names": full_dataset.classes,
}, "Model/fashion_mood_model_pytorch.pth")
```

## 2. 웹앱 실행

프로젝트 폴더의 PowerShell에서 실행한다.

```powershell
.\.venv-pytorch-gpu\Scripts\python.exe -m streamlit run app.py
```

이미지를 업로드하면 분석 패널에 실제 무드와 예측 점수가 표시된다. CUDA를
사용할 수 있으면 GPU로 추론하고, 없으면 CPU로 추론한다. 추론에서는 eval 모드와
inference_mode를 사용한다. 성별·의류 종류는 현재 무드 모델의 예측 대상이 아니므로
아래 필터에서 선택한다.

학습 파일이 아직 없거나 저장이 덜 끝났다면 안내가 표시된다. 저장 완료 후
`모델 다시 확인`을 누르거나 페이지를 새로고침한다. 파일 변경도 캐시 키에 반영된다.
같은 업로드의 필터·상세보기 조작 시에는 세션에 저장한 분석을 재사용한다.

단일 이미지 예측만 확인하려면 다음과 같이 실행할 수 있다.

```powershell
.\.venv-pytorch-gpu\Scripts\python.exe model.py --image "사진경로.jpg"
```

## 3. 실제 유사도 TOP 5를 위한 후보 인덱스 만들기

무드 분류 결과만으로는 특정 이미지 파일을 찾을 수 없다. 웹에서 추천할 후보의
특징 벡터를 먼저 저장해야 한다. **학습용 전체 데이터가 아닌 웹 추천 카탈로그**를
입력한다. 이 작업은 사용자가 아래 명령을 실행할 때만 수행하며 앱에서 전체
데이터셋을 자동으로 처리하지 않는다.

앱의 카탈로그 우선순위는 `recommend_pool/metadata_web.csv` → `metadata_r2.csv`
→ `metadata.csv` → 프로젝트 루트의 `metadata.csv`다. 존재하는 파일 중 앱에서
사용하는 것과 동일한 CSV를 명령에 지정한다. 후보 이미지는 실행 PC에서
`original_img_path`로 접근할 수 있어야 한다. 상대 경로는 프로젝트 폴더 기준이다.

```powershell
.\.venv-pytorch-gpu\Scripts\python.exe model.py --build-index recommend_pool/metadata.csv
```

학습 중인 GPU 사용을 피하려면 다음과 같이 CPU를 지정할 수 있다. 모델 가중치는
이미 저장되어 있어야 한다.

```powershell
.\.venv-pytorch-gpu\Scripts\python.exe model.py --build-index recommend_pool/metadata.csv --device cpu
```

결과 파일:

- `recommend_pool/item_embeddings_web.npy`: CSV와 동일한 행 순서의 256차원 벡터.
- `recommend_pool/item_embeddings_web.json`: 모델·전처리·카탈로그·벡터 파일 검증 정보.

앱은 이 인덱스로 코사인 유사도를 계산한다. 인덱스가 없거나 다른 모델로 만든
파일이면 일반 카탈로그 탐색으로 표시하며 유사도 순위라고 안내하지 않는다.
가중치나 후보 목록·행 순서가 바뀌면 인덱스를 다시 생성한다. 다른 PC에는 벡터와
JSON을 함께 전달한다.

현재 프로젝트에 카탈로그 CSV가 없다면 JSON에서 먼저 생성해야 한다.
`preprocess.py`의 `--json-dir`, `--image-dir`, `--output` 옵션을 사용할 수 있다.
전체 학습 폴더를 대상으로 명령을 실행하면 그 전체가 카탈로그가 되므로,
웹용으로 선정한 후보 범위를 먼저 정하는 것이 좋다.
