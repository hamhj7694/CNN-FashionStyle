# GPU 학습 환경

이 프로젝트는 모델별 GPU 의존성이 달라 두 개의 Jupyter 커널을 사용한다.

## PyTorch 노트북

대상 파일: `CNN_model_PyTorch.ipynb`

- Python 3.12
- PyTorch 2.6.0
- torchvision 0.21.0
- CUDA 12.4 wheel
- 커널: `Python (MoodFit PyTorch GPU)`

검증 결과 RTX 3060에서 다음 상태로 정상 동작한다.

```text
torch 2.6.0+cu124
CUDA 12.4
cuda_available=True
NVIDIA GeForce RTX 3060
```

## TensorFlow 노트북

대상 파일: `CNN_model.ipynb`

- Python 3.10
- TensorFlow 2.10.0
- CUDA Toolkit 11.2
- cuDNN 8.1
- 커널: `Python (MoodFit TensorFlow GPU)`

Windows 네이티브에서 TensorFlow GPU를 사용하려면 TensorFlow 2.10 이하가 필요하다. TensorFlow 2.11 이상은 Windows 네이티브 CUDA GPU를 지원하지 않으며 WSL2가 필요하다.

현재 Python 패키지는 설치되어 있지만 다음 DLL이 없어 TensorFlow GPU가 아직 활성화되지 않았다.

```text
cudart64_110.dll
cublas64_11.dll
cudnn64_8.dll
```

따라서 NVIDIA CUDA Toolkit 11.2와 cuDNN 8.1을 설치하고, 설치 후 TensorFlow 커널을 재시작한다. 검증 명령은 다음과 같다.

```powershell
\.venv-tensorflow-gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 커널 선택

`CNN_model_PyTorch.ipynb`에서는 `Python (MoodFit PyTorch GPU)`를 선택하고, `CNN_model.ipynb`에서는 `Python (MoodFit TensorFlow GPU)`를 선택한다.

두 노트북 모두 GPU를 찾지 못하면 학습을 시작하지 않고 오류를 발생시키도록 설정되어 있다.
