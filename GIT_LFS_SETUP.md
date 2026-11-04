# 다른 PC에서 MoodFit AI 작업 이어가기

이 문서는 GitHub와 Git LFS에 저장된 MoodFit AI 코드, 학습 이미지, JSON 라벨을 다른 PC에서 내려받아 실행하는 방법을 설명한다.

## 1. 필요한 프로그램

다른 PC에 다음 프로그램을 먼저 설치한다.

- Git for Windows: <https://git-scm.com/download/win>
- Git LFS: <https://git-lfs.com/>
- Python 3.10 이상 권장

Git LFS 설치 후 PowerShell에서 정상 설치 여부를 확인한다.

```powershell
git --version
git lfs version
```

## 2. 저장소 내려받기

PowerShell에서 작업할 폴더로 이동한 뒤 저장소를 clone한다.

```powershell
cd C:\Users\$env:USERNAME\Documents\GitHub
git clone https://github.com/hamhj7694/CNN-FashionStyle.git
cd .\CNN-FashionStyle
```

Git LFS를 현재 저장소에 활성화한다.

```powershell
git lfs install
```

## 3. LFS 이미지 내려받기

코드와 Git LFS 포인터만 내려온 경우 다음 명령을 실행한다.

```powershell
git lfs pull
```

전체 학습 이미지가 많기 때문에 다운로드에 시간이 걸릴 수 있다. 다운로드 상태를 확인하려면 다음 명령을 사용한다.

```powershell
git lfs status
git lfs ls-files | Select-Object -First 10
```

이미지가 아직 포인터 파일로 남아 있으면 다음 명령을 순서대로 실행한다.

```powershell
git lfs fetch origin main
git lfs checkout
```

## 4. 이미지 다운로드 검증

다음 파일이 실제 JPG 이미지로 존재하는지 확인한다.

```powershell
$sample = Join-Path (Get-Location) 'CNN_train_test\CNN_train_test_image\밀리터리\3725.jpg'
Get-Item $sample | Select-Object FullName, Length
```

파일 크기가 수 KB 이상이고, 아래 Python 코드가 오류 없이 실행되면 정상이다.

```powershell
python -c "from PIL import Image; p=r'CNN_train_test\CNN_train_test_image\밀리터리\3725.jpg'; im=Image.open(p); print(im.format, im.size)"
```

JSON 라벨도 함께 확인한다.

```powershell
Test-Path '.\CNN_train_test\CNN_train_test_label\밀리터리\3725.json'
```

## 5. Python 환경 구성

저장소 루트에서 가상환경을 만든다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell 실행 정책 오류가 발생하면 현재 사용자 범위에서 한 번만 허용한다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 6. JSON 메타데이터 생성

AI Hub JSON과 이미지의 실제 경로를 이용해 `recommend_pool/metadata.csv`를 생성한다.

```powershell
python preprocess.py `
  --json-dir CNN_train_test/CNN_train_test_label `
  --image-dir CNN_train_test/CNN_train_test_image `
  --output recommend_pool/metadata.csv
```

이 단계에서는 원본 이미지를 미리 크롭하지 않는다. JSON의 bbox 정보만 CSV에 저장하고, 추천 화면에 표시될 때 필요한 크롭만 생성한다.

## 7. Streamlit 실행

```powershell
streamlit run app.py
```

브라우저에서 다음 주소를 연다.

```text
http://localhost:8501
```

## 8. LFS 용량 주의사항

현재 `CNN_train_test`는 약 7GiB 규모이며 JPG 약 12만 개가 포함되어 있다. GitHub Free의 Git LFS 기본 저장공간·다운로드 대역폭은 각각 월 10GiB이므로, 다른 PC에서 전체 이미지를 내려받으면 대역폭을 대부분 사용한다.

코드만 먼저 내려받고 이미지는 나중에 받고 싶다면 clone 전에 LFS 자동 다운로드를 끈다.

이미지가 필요 없는 PC에서는 LFS 파일을 내려받지 않고 작업할 수 있다.

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/hamhj7694/CNN-FashionStyle.git
```

이 경우 이미지가 필요한 시점에 저장소 폴더에서 다음 명령을 실행한다.

```powershell
Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
git lfs pull
```

## 9. 작업 후 변경사항 업로드

코드와 문서만 변경한 경우:

```powershell
git add app.py model.py preprocess.py curation.py PRD.md GIT_LFS_SETUP.md requirements.txt .gitattributes .gitignore
git commit -m "Update MoodFit AI pipeline and LFS setup"
git push origin main
```

학습 이미지나 JSON을 추가한 경우에도 Git LFS 규칙이 적용된 이미지라면 일반 Git과 같은 방식으로 추가한다.

```powershell
git add CNN_train_test
git commit -m "Add training dataset"
git push origin main
```

이미지 파일은 `.gitattributes`의 LFS 규칙에 따라 실제 바이너리 대신 Git에는 포인터가 저장되고, 실제 내용은 Git LFS 저장소에 업로드된다.

