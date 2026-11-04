"""학습된 ConvNeXt-Tiny를 사용하는 MoodFit 추론 인터페이스.

CNN_model_PyTorch.ipynb의 state_dict, 분류기, 전처리 및 클래스 순서를
그대로 사용한다. import 시점에는 모델을 읽거나 다운로드하지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "Model" / "fashion_mood_model_pytorch4_street_cap_10000_more_layers_best.pth"
# ImageFolder.classes 및 사용자의 불러오기 코드와 같은 순서다.
CLASS_NAMES = (
    "기타", "레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시",
    "소피스트케이티드", "스트리트", "스포티", "아방가르드", "오리엔탈", "웨스턴",
    "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피",
    "히피", "힙합",
)
EMBEDDING_DIM = 256
EMBEDDING_VERSION = "convnext-tiny-head256-resize224-imagenet-l2-v1"


def get_model_signature(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[str, int, int]:
    """파일이 새로 생기거나 바뀌면 Streamlit 캐시 키도 바뀐다."""
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return str(path), 0, 0
    return str(path), stat.st_mtime_ns, stat.st_size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_extractor(
    model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None,
) -> Any:
    """저장된 state_dict를 읽어 eval 상태의 PyTorch 모델을 반환한다.

    CUDA가 있으면 GPU, 없으면 CPU를 사용한다. device="cpu"로 지정하면
    학습 중인 GPU를 사용하지 않고 웹 추론이나 인덱스 생성을 할 수 있다.
    원래 노트북의 state_dict 또는 state_dict/model_state_dict 키를 가진
    checkpoint를 지원한다. class_names가 있으면 함께 읽는다.
    """
    signature = get_model_signature(model_path)
    path = Path(signature[0])
    if not path.is_file():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {path}. 학습 완료 후 mood_model.state_dict()를 저장해 주세요."
        )
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "PyTorch/torchvision을 불러올 수 없습니다. .venv-pytorch-gpu 환경에서 앱을 실행해 주세요."
        ) from exc

    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU를 사용할 수 없습니다. device='cpu' 또는 CUDA 환경을 확인하세요.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("가중치를 읽지 못했습니다. 모델 저장이 끝난 .pth 파일인지 확인해 주세요.") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("torch.save(mood_model.state_dict(), model_path)로 저장한 파일이 필요합니다.")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    if (not isinstance(class_names, (list, tuple)) or not class_names
            or any(not isinstance(name, str) or not name for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("checkpoint의 class_names는 중복 없는 클래스명 목록이어야 합니다.")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint의 state_dict 형식이 올바르지 않습니다.")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

    network = models.convnext_tiny(weights=None)
    in_features = network.classifier[2].in_features
    network.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_features, EMBEDDING_DIM),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(EMBEDDING_DIM, len(class_names)),
    )
    try:
        network.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "가중치와 모델 구조가 맞지 않습니다. 학습 모델의 Flatten → Linear(768,256) → "
            "ReLU → Dropout(0.3) → Linear(256,24) 구조 및 클래스 순서를 확인해 주세요."
        ) from exc

    network.to(target)
    network.eval()
    network.requires_grad_(False)
    network.class_names = tuple(class_names)
    network.preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    network.checkpoint_path = path
    network.checkpoint_sha256 = _sha256(path)
    if get_model_signature(path) != signature:
        raise ValueError("모델 파일이 저장 중 변경되었습니다. 저장 완료 후 다시 시도해 주세요.")
    return network


def _rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if not np.isfinite(image).all():
            raise ValueError("이미지 배열에 유효하지 않은 값이 있습니다.")
        if image.dtype != np.uint8:
            if image.size and (image.min() < 0 or image.max() > 255):
                raise ValueError("이미지 배열은 0~255 또는 0~1 범위여야 합니다.")
            image = (image * 255 if image.size and image.max() <= 1 else image).astype(np.uint8)
        return _rgb_image(Image.fromarray(image))
    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    with Image.open(image) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def analyze_image(image: Any, model: Any) -> dict[str, Any]:
    """한 번의 순전파로 무드 확률과 L2 정규화된 256차원 벡터를 반환한다."""
    import torch

    tensor = model.preprocess(_rgb_image(image)).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)
    with torch.inference_mode():
        features = model.avgpool(model.features(tensor))
        for layer in model.classifier[:-1]:
            features = layer(features)
        probabilities = torch.softmax(model.classifier[-1](features), dim=1)[0]
        scores = probabilities.cpu().numpy().astype(np.float32)
        embedding = features[0].cpu().numpy().astype(np.float32)
    if not np.isfinite(scores).all() or not np.isfinite(embedding).all():
        raise ValueError("추론 결과에 유효하지 않은 값이 있습니다. 모델 가중치를 확인해 주세요.")
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    prediction = int(np.argmax(scores))
    return {
        "mood": model.class_names[prediction],
        "confidence": float(scores[prediction]),
        "mood_scores": dict(zip(model.class_names, map(float, scores))),
        "embedding": embedding,
    }


def extract_image_embedding(image: Any, model: Any) -> np.ndarray:
    """기존 앱·큐레이션 계약을 유지하는 추천용 특징 추출 함수."""
    return analyze_image(image, model)["embedding"]


def predict_mood(image: Any, model: Any) -> dict[str, Any]:
    """예측 무드·점수·클래스별 확률을 반환한다. 성별·의류 부위는 추론하지 않는다."""
    result = analyze_image(image, model)
    return {key: value for key, value in result.items() if key != "embedding"}


def catalog_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """벡터와 카탈로그의 행 순서가 같은지 확인한다."""
    rows = [
        [str(row.get("id", "")), str(row.get("original_img_path", row.get("image_path", "")))]
        for row in records
    ]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_embedding_index(
    metadata_path: str | Path, output_path: str | Path | None = None,
    *, model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None,
) -> Path:
    """웹 카탈로그와 동일한 행 순서로 후보 벡터와 검증용 manifest를 저장한다.

    이미지가 없으면 실패한 행을 알려주며 중간 행을 생략하지 않는다.
    학습 데이터 전체가 아니라 웹 추천 카탈로그를 입력한다.
    """
    import pandas as pd
    from tqdm.auto import tqdm

    metadata_path = Path(metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    frame = pd.read_csv(metadata_path).fillna("")
    if "original_img_path" not in frame.columns and "image_path" in frame.columns:
        frame = frame.rename(columns={"image_path": "original_img_path"})
    if frame.empty or "original_img_path" not in frame.columns:
        raise ValueError("이미지가 있는 카탈로그 CSV와 original_img_path 컬럼이 필요합니다.")
    destination = Path(output_path) if output_path else ROOT / "recommend_pool" / "item_embeddings_web.npy"
    if not destination.is_absolute():
        destination = ROOT / destination
    if destination.suffix != ".npy":
        raise ValueError("벡터 출력 경로는 .npy 확장자여야 합니다.")
    network = load_feature_extractor(model_path, device=device)
    records = frame.to_dict("records")
    embeddings = np.empty((len(records), EMBEDDING_DIM), dtype=np.float32)
    for position, record in enumerate(tqdm(records, desc="웹 후보 특징 추출", unit="image")):
        source = Path(str(record["original_img_path"]))
        if not source.is_absolute():
            source = ROOT / source
        try:
            embeddings[position] = extract_image_embedding(source, network)
        except (OSError, ValueError) as exc:
            raise ValueError(f"카탈로그 행 {position}, ID={record.get('id', '')}: {source}") from exc
    manifest = {
        "embedding_version": EMBEDDING_VERSION,
        "checkpoint_sha256": network.checkpoint_sha256,
        "catalog_fingerprint": catalog_fingerprint(records),
        "shape": list(embeddings.shape),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npy.tmp")
    with temporary.open("wb") as target:
        np.save(target, embeddings, allow_pickle=False)
    manifest["embedding_sha256"] = _sha256(temporary)
    temporary.replace(destination)
    manifest_path = destination.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoodFit 저장 모델 추론 및 웹 후보 인덱스 생성")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default=None)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--image", type=Path, help="단일 이미지의 무드 예측")
    action.add_argument("--build-index", type=Path, metavar="METADATA_CSV", help="웹 카탈로그 CSV")
    parser.add_argument("--output", type=Path, help="후보 벡터 저장 경로 (.npy)")
    args = parser.parse_args()
    if args.image:
        loaded = load_feature_extractor(args.model, device=args.device)
        result = predict_mood(args.image, loaded)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        saved = build_embedding_index(args.build_index, args.output, model_path=args.model, device=args.device)
        print(f"추천 인덱스 저장 완료: {saved}")
