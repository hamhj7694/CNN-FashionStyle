# 인터넷 의상 사진 10장 테스트

24개 학습 무드 중 10가지를 선정해 Wikimedia Commons의 사진을 내려받았다.
`reference_mood`는 사진 선정 시 붙인 참고 무드이며 AI Hub의 공식 정답이 아니다.
스타일은 겹칠 수 있으므로 이 사진들과의 일치율을 모델의 정식 정확도로 해석하지 않는다.
레트로 예시는 마네킹 드레스, 밀리터리 예시는 재킷 단독 사진으로, 착장 사진과의 입력 차이도 있다.

## 저장된 파일

- `images/`: 이미지 10장. 노트북의 `predict_image()`에 로컬 경로로 전달할 수 있다.
- `samples.json`: 참고 무드, 원본 페이지·이미지 URL, 저작자와 라이선스.
- `predictions.csv`: 실제 저장 모델의 예측 무드, softmax 점수, 상위 3개 결과 및 가중치 체크섬.
- `report.html`: 사진과 예측을 함께 보는 로컬 결과 페이지.

프로젝트 폴더의 PowerShell에서 결과 페이지를 열 수 있다.

```powershell
Invoke-Item ".\test_assets\internet_outfits\report.html"
```

## 저장 모델로 다시 테스트

이번 실행에 사용한 파일은 `Model/fashion_mood_model_pytorch.pth`다.
웹앱의 기본 경로 `Model/fashion_mood_model.pth`와 이름이 다르므로 혼동하지 않는다.
기존 노트북과 `mood_model_가져오기.py`는 수정하지 않았다.

```powershell
.\.venv-pytorch-gpu\Scripts\python.exe -X utf8 scripts/test_internet_outfits.py --no-download --model Model/fashion_mood_model_pytorch.pth --device cpu
```

CPU 추론은 학습 중인 GPU와 경쟁하지 않도록 선택했다. 학습이 끝난 뒤 GPU로
테스트하려면 `--device cuda:0`으로 바꾼다. 사진을 처음부터 내려받으려면
`--no-download` 옵션을 빼고 실행한다. 기존 사진은 재사용한다.

## 기존 노트북에서 직접 테스트

`def predict_image(img_path): ...`는 함수를 정의하는 코드다. 정의만 실행해서는
예측이 출력되지 않는다. 학습을 마친 뒤 함수를 호출해야 한다.

한 장을 확인하려면:

```python
predict_image("test_assets/internet_outfits/images/01_스트리트.jpg")
```

10장을 사진과 함께 확인하려면 새 셀에서 다음 코드를 실행한다. 이 방법은
저장 파일을 다시 읽는 대신 노트북 메모리에 있는 현재 `mood_model`을 사용한다.
따라서 저장 시점과 메모리 모델의 가중치가 다르면 CSV와 예측이 달라질 수 있다.

```python
import json
from pathlib import Path
from IPython.display import Image as DisplayImage, display

sample_root = Path("test_assets/internet_outfits")
samples = json.loads((sample_root / "samples.json").read_text(encoding="utf-8"))

for sample in samples:
    image_path = sample_root / sample["local_path"]
    print(f"\n선정 참고 무드: {sample['reference_mood']}")
    display(DisplayImage(filename=str(image_path), width=240))
    predict_image(str(image_path))
```

`Image.open()`은 일반 웹페이지나 이미지 URL을 직접 내려받지 않는다. 인터넷 주소는
먼저 이미지 바이트를 다운로드한 뒤 열거나, 이번처럼 저장된 이미지 경로를 전달한다.

## 출처

각 사진의 원본 페이지와 라이선스는 `samples.json`, `predictions.csv`, `report.html`에
함께 남겼다. 원본 링크는 결과 페이지의 `원본·출처`에서 확인할 수 있다.
