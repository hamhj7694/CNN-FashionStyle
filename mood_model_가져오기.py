"""V4 최고 모델을 불러와 이미지 한 장의 무드를 예측하는 예제 스크립트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import model as moodfit


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "Model" / "fashion_mood_model_pytorch4_street_cap_10000_more_layers_best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="MoodFit V4 이미지 무드 예측")
    parser.add_argument("image", type=Path, help="예측할 이미지 경로")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default=None)
    args = parser.parse_args()

    network = moodfit.load_feature_extractor(args.model, device=args.device)
    result = moodfit.predict_mood(args.image, network)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
