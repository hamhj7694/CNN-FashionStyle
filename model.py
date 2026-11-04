"""MoodFit AI 모델 연동 인터페이스.

학습과 실제 TensorFlow/Keras 모델 선택은 사용자 구현 영역이다. 앱과 추천
파이프라인은 아래 두 함수를 통해서만 모델을 호출하도록 설계되어 있다.
"""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf  # noqa: F401  # 사용자 구현 시 사용
except ImportError:  # TensorFlow가 없는 환경에서도 전처리/데모 UI는 실행 가능
    tf = None


def load_feature_extractor() -> Any:
    """TensorFlow/Keras feature extractor를 반환한다.

    실제 모델을 연결할 때 이 함수에서 ``tf.keras.models.load_model`` 또는
    원하는 backbone을 반환하면 된다.
    """

    raise NotImplementedError(
        "load_feature_extractor()에 사용할 TensorFlow/Keras 모델을 구현해 주세요."
    )


def extract_image_embedding(image: Any, model: Any) -> Any:
    """이미지에서 특징 벡터를 추출한다.

    Args:
        image: PIL Image, numpy array 등 모델이 처리할 이미지 입력.
        model: ``load_feature_extractor``가 반환한 모델.
    """

    raise NotImplementedError(
        "extract_image_embedding(image, model)에 전처리와 model.predict를 구현해 주세요."
    )
