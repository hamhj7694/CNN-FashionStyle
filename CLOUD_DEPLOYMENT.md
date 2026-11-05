# MoodFit AI 클라우드 실행

앱 실행은 Streamlit Community Cloud, 이미지 저장은 Cloudflare R2로 분리합니다.

## 1. 웹용 이미지 생성

프로젝트 루트에서 Python 3.12 가상환경으로 실행합니다. 원본은 변경하지 않으며, 긴 변을 최대 1,280px로 줄인 WebP(quality 80) 사본을 생성합니다. 같은 원본을 사용하는 annotation은 웹 이미지 하나를 공유합니다.

```powershell
python prepare_web_images.py --workers 8
```

- 이미지: `recommend_pool/web_assets/`
- 준비된 목록: `recommend_pool/metadata_optimized.csv`
- 진행률: `recommend_pool/metadata_optimized.progress.json`
- 읽지 못한 이미지: `recommend_pool/metadata_optimized.errors.json` (발생한 경우)

`--asset-dir`와 `--output`으로 저장 위치를 바꿀 수 있습니다. 중단 후 같은 명령을 실행하면 이미 완료된 파일을 재사용합니다. 첫 시험은 `--limit 20`으로 수행할 수 있습니다. 최종 배포 전에는 limit 없이 전체 목록을 생성합니다.

현재 웹 UI는 전신 이미지를 카드와 선택 뷰어에서 사용하므로 이 경로는 전신 WebP만 업로드합니다. 개별 의류 bbox 크롭까지 필요하면 기존 `upload_crops_to_r2.py` 경로를 별도로 사용합니다.

## 2. R2 연결 및 업로드

Cloudflare에서 R2 버킷을 생성하고 공개 이미지 URL을 준비합니다. 개발 확인은 `r2.dev`, 지속적인 공개 서비스는 커스텀 도메인 구성을 사용합니다.

`.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사한 뒤 R2 계정 ID, 버킷 이름, 공개 URL, 해당 버킷의 객체 읽기/쓰기 자격 증명을 설정합니다. 실제 비밀키를 Git에 커밋하지 않습니다.

먼저 파일·용량만 확인합니다. dry run은 자격 증명 없이 실행할 수 있고 공개 CSV를 변경하지 않습니다.

```powershell
python upload_web_images.py --dry-run
```

실제 업로드:

```powershell
python upload_web_images.py --workers 8
```

중단 시 `.uploads.json` 체크포인트를 사용해 업로드를 재개합니다. 모든 준비된 이미지 업로드가 끝나야 `recommend_pool/metadata_r2.csv`를 교체합니다. 이 CSV에는 브라우저가 접근할 `full_image_url`을 넣고, 준비 PC의 `full_image_path`는 제거합니다.

공개 URL로 WebP가 열리는지 확인한 뒤 `metadata_r2.csv`를 Git에 반영합니다. 앱은 `metadata_web.csv`가 있으면 먼저 선택하므로, 배포할 카탈로그가 무엇인지 확인하세요.

## 3. Streamlit Community Cloud 설정

공개 GitHub 저장소에서 다음 값을 지정합니다.

| 항목 | 값 |
| --- | --- |
| Repository | `hamhj7694/CNN-FashionStyle` |
| Branch | `main` |
| Main file path | `deploy/streamlit_app.py` |
| Python | `3.12` |

`deploy/requirements.txt`를 entrypoint 옆에 두어 CPU용 PyTorch를 설치합니다. 로컬 학습용 CUDA 의존성과 Jupyter 전체를 배포 서버에 설치하지 않습니다.

현재 기본 V4 모델은 Git LFS 파일입니다. 배포 환경에서 실제 가중치를 내려받았는지 확인합니다. 공개 R2 URL을 조회하는 앱에는 업로드용 R2 비밀키를 넣을 필요가 없습니다.

인덱스가 없으면 조건별 추천으로 실행되며, 코사인 유사도 추천을 사용하려면 배포 카탈로그와 일치하는 벡터·검증 JSON을 별도로 준비합니다.

## 4. 배포 후 확인

1. 첫 화면과 모든 추천 이미지가 실제로 표시되는지 확인합니다.
2. 사진 업로드 후 진단·첫 추천 선택·다른 카드 전환을 확인합니다.
3. 무드·의류 분류를 바꾸고 결과 개수와 초기화를 확인합니다.
4. 좌우 이미지의 휠 줌·드래그를 확인합니다.
5. 공개 앱 URL을 README의 라이브 데모 링크로 추가합니다.

Community Cloud는 비활성 앱이 절전될 수 있으며, CPU 추론 속도와 동시 접속은 배포 자원에 따라 달라집니다.
