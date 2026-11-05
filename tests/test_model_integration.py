"""실제 ConvNeXt 구조와 임시 가중치로 수행하는 CPU 연동 테스트."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models

import model


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 실제 학습 중인 GPU와 경쟁하지 않도록 CPU와 소수 스레드만 사용한다.
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory(prefix="moodfit-integration-")
        cls.root = Path(cls.directory.name)
        cls.path = cls.root / "test_weights.pth"
        source = models.convnext_tiny(weights=None)
        source.classifier = nn.Sequential(
            nn.Flatten(1), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 24),
        )
        torch.save(source.state_dict(), cls.path)
        cls.network = model.load_feature_extractor(cls.path, device="cpu")
        cls.image = Image.new("RGB", (150, 280), (130, 50, 180))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_network_and_inference_match(self):
        self.assertFalse(self.network.training)
        self.assertTrue(all(not p.requires_grad for p in self.network.parameters()))
        self.assertEqual(self.network.class_names, model.CLASS_NAMES)
        result = model.analyze_image(self.image, self.network)
        tensor = self.network.preprocess(self.image).unsqueeze(0)
        with torch.inference_mode():
            expected = torch.softmax(self.network(tensor), dim=1)[0].numpy()
        np.testing.assert_allclose(list(result["mood_scores"].values()), expected, rtol=1e-6)
        self.assertEqual(result["mood"], model.CLASS_NAMES[int(expected.argmax())])
        self.assertEqual(result["embedding"].shape, (256,))
        self.assertEqual(result["embedding"].dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(result["embedding"])), 1.0, places=5)
        self.assertAlmostEqual(sum(result["mood_scores"].values()), 1.0, places=5)

    def test_public_functions_and_image_formats(self):
        image_bytes = io.BytesIO()
        self.image.save(image_bytes, format="PNG")
        reference = model.extract_image_embedding(self.image, self.network)
        np.testing.assert_allclose(
            model.extract_image_embedding(image_bytes.getvalue(), self.network), reference,
        )
        np.testing.assert_allclose(
            model.extract_image_embedding(np.asarray(self.image), self.network), reference,
        )
        self.assertEqual(len(model.predict_mood(self.image.convert("RGBA"), self.network)["mood_scores"]), 24)

    def test_missing_corrupt_and_mismatched_checkpoints(self):
        with self.assertRaises(FileNotFoundError):
            model.load_feature_extractor(self.root / "missing.pth", device="cpu")
        bad = self.root / "bad.pth"
        bad.write_bytes(b"unfinished checkpoint")
        with self.assertRaisesRegex(ValueError, "가중치를 읽지"):
            model.load_feature_extractor(bad, device="cpu")
        torch.save({"classifier.4.weight": torch.zeros(3, 256)}, bad)
        with self.assertRaisesRegex(ValueError, "모델 구조"):
            model.load_feature_extractor(bad, device="cpu")

    def test_signature_updates_when_file_appears(self):
        path = self.root / "signature.pth"
        missing = model.get_model_signature(path)
        path.write_bytes(b"saved")
        self.assertNotEqual(model.get_model_signature(path), missing)
        self.assertEqual(model.get_model_signature(path)[2], 5)

    def test_catalog_index_keeps_row_mapping_and_manifest(self):
        first = self.root / "first.jpg"
        second = self.root / "second.jpg"
        self.image.save(first)
        Image.new("RGB", (180, 100), "green").save(second)
        records = [
            {"id": "b", "original_img_path": str(second), "mood": "모던"},
            {"id": "a", "original_img_path": str(first), "mood": "로맨틱"},
        ]
        metadata = self.root / "metadata.csv"
        pd.DataFrame(records).to_csv(metadata, index=False)
        output = self.root / "embeddings.npy"
        with patch("model.load_feature_extractor", return_value=self.network):
            model.build_embedding_index(metadata, output)
        matrix = np.load(output, allow_pickle=False)
        self.assertEqual(matrix.shape, (2, 256))
        np.testing.assert_allclose(matrix[0], model.extract_image_embedding(second, self.network))
        manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["checkpoint_sha256"], self.network.checkpoint_sha256)
        self.assertEqual(manifest["catalog_fingerprint"], model.catalog_fingerprint(records))
        self.assertNotEqual(model.catalog_fingerprint(records), model.catalog_fingerprint(records[::-1]))
        self.assertEqual(manifest["embedding_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_streamlit_index_validation_and_similarity_mapping(self):
        from streamlit.testing.v1 import AppTest

        index_path = self.root / "validated_index.npy"
        matrix = np.zeros((2, 256), dtype=np.float32)
        matrix[0, 0] = 1
        matrix[1, 1] = 1
        np.save(index_path, matrix, allow_pickle=False)
        records = [
            {"id": "first", "original_img_path": "first.jpg", "mood": "모던"},
            {"id": "second", "original_img_path": "second.jpg", "mood": "모던"},
        ]
        manifest = {
            "embedding_version": model.EMBEDDING_VERSION,
            "checkpoint_sha256": "test-checkpoint",
            "catalog_fingerprint": model.catalog_fingerprint(records),
            "embedding_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "shape": [2, 256],
        }
        index_path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
        script = f'''
from pathlib import Path
import pandas as pd
import app
import model
from curation import get_recommendation_sections
path = Path({str(index_path)!r})
records = {records!r}
fingerprint = model.catalog_fingerprint(records)
matrix, error = app._load_embedding_file(str(path), (1,), 2, "test-checkpoint", fingerprint)
assert matrix is not None and not error
sections = get_recommendation_sections(matrix[1], "전체", pd.DataFrame(records), matrix)
assert sections["similar"][0]["id"] == "second"
assert sections["similar"][0]["similarity"] == 1.0
assert not ({{r["id"] for r in sections["similar"]}} & {{r["id"] for r in sections["discovery"]}})
assert app._load_embedding_file(str(path), (1,), 2, "different-checkpoint", fingerprint)[0] is None
assert app._load_embedding_file(str(path), (1,), 2, "test-checkpoint", "different-order")[0] is None
assert app._load_embedding_file(str(path), (1,), 3, "test-checkpoint", fingerprint)[0] is None
with path.open("ab") as target:
    target.write(b"changed")
assert app._load_embedding_file(str(path), (2,), 2, "test-checkpoint", fingerprint)[0] is None
'''
        tested = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(tested.exception), 0)

    def test_streamlit_initial_screen(self):
        from streamlit.testing.v1 import AppTest

        tested = AppTest.from_file(str(model.ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(tested.exception), 0)

    def test_streamlit_prediction_and_missing_model_panels(self):
        from streamlit.testing.v1 import AppTest

        script = f'''
import io
from PIL import Image
import streamlit as st
import app
import model
app.HAS_SHADCN = False
network = model.load_feature_extractor({str(self.path)!r}, device="cpu")
upload = io.BytesIO()
Image.new("RGB", (128, 150), "blue").save(upload, format="PNG")
result, error = app.analyze_upload(upload, network)
assert not error
app.render_analysis_panel(True, result)
assert app.analyze_upload(upload, network)[0] is result
app.render_analysis_panel(True, None, "학습 완료 후 모델 저장이 필요합니다.")
'''
        tested = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(tested.exception), 0)
        markdown = "\n".join(item.value for item in tested.markdown)
        self.assertIn("예측 점수", markdown)
        self.assertNotIn("Modern Minimal", markdown)
        self.assertTrue(any("모델 저장" in item.value for item in tested.info))


if __name__ == "__main__":
    unittest.main()
