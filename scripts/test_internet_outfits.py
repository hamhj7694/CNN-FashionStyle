"""인터넷 의상 사진 10개를 내려받아 저장된 MoodFit 모델로 예측한다.

python scripts/test_internet_outfits.py --download-only
python scripts/test_internet_outfits.py --no-download --device cpu

선정 무드는 사람이 붙인 참고값이며 AI Hub의 공식 정답 라벨이 아니다.
학습 노트북과 모델 불러오기 연습 파일은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model as moodfit


OUTPUT = ROOT / "test_assets" / "internet_outfits"
USER_AGENT = "MoodFitAI-EducationalTest/1.0 (local fashion classifier experiment)"
# 정답 검증 세트가 아니라 서로 다른 스타일을 살펴보기 위한 외부 예시다.
SAMPLES = [
    ("01", "스트리트", "Street fashion 2019 .jpg", "스트리트 스타일 착장"),
    ("02", "프레피", "Preppy,.jpg", "프레피 스타일 착장"),
    ("03", "히피", "Conjunto-bolero-falda.jpg", "히피 시크 타이다이 착장"),
    ("04", "펑크", "Modpunk07.jpg", "펑크 스타일 착장"),
    ("05", "스포티", "FashionTracksuit.jpg", "트레이닝복 착장"),
    ("06", "클래식", "Gentleman in bowler.JPG", "정장과 중절모 착장"),
    ("07", "웨스턴", "Cowgirl model.jpg", "카우걸 스타일 착장"),
    ("08", "레트로", "1950s vintage dress.jpg", "1950년대 빈티지 드레스"),
    ("09", "페미닌", "Woman pink and blue dress.jpg", "분홍·파랑 꽃무늬 드레스"),
    ("10", "밀리터리", "M-65 field jacket.jpg", "M-65 필드 재킷"),
]


def _download(url: str, limit: int = 16 * 1024 * 1024) -> bytes:
    if urlsplit(url).scheme != "https":
        raise ValueError("HTTPS 이미지 링크를 사용해 주세요.")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("테스트 이미지 또는 응답이 너무 큽니다.")
    return data


def _plain(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", text)).split())


def download_samples(output: Path = OUTPUT, refresh: set[str] | None = None) -> list[dict[str, str]]:
    """출처·저작자·라이선스와 함께 이미지 10개를 로컬에 저장한다."""
    query = {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "960",
        "titles": "|".join("File:" + sample[2] for sample in SAMPLES),
    }
    response = json.loads(_download("https://commons.wikimedia.org/w/api.php?" + urlencode(query)))
    pages = {page["title"]: page for page in response["query"]["pages"]}
    normalized = {entry["from"]: entry["to"] for entry in response["query"].get("normalized", [])}
    records = []
    for sample_id, mood, filename, description in SAMPLES:
        title = "File:" + filename
        page = pages.get(normalized.get(title, title))
        if not page or not page.get("imageinfo"):
            raise ValueError(f"원본 사진을 찾지 못했습니다: {title}")
        info = page["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        # 이미 수백 KB~수 MB인 JPEG 원본을 사용한다. 새 썸네일 생성에 따른
        # Wikimedia의 429 제한을 피하고 추가 가공 없이 테스트한다.
        url = info["url"]
        split = urlsplit(url)
        url = urlunsplit((split.scheme, split.netloc, split.path, "", ""))
        records.append({
            "id": sample_id, "reference_mood": mood, "description": description,
            "source_title": page["title"], "source_url": info["descriptionurl"],
            "image_url": url, "local_path": f"images/{sample_id}_{mood}.jpg",
            "author": _plain(metadata.get("Artist", {}).get("value", "")),
            "license": _plain(metadata.get("LicenseShortName", {}).get("value", "")),
            "license_url": metadata.get("LicenseUrl", {}).get("value", ""),
            "note": "선정 무드는 참고값이며 공식 정답 라벨이 아닙니다.",
        })
    (output / "images").mkdir(parents=True, exist_ok=True)

    def save(record: dict[str, str]) -> None:
        path = output / record["local_path"]
        if path.is_file() and record["id"] not in (refresh or set()):
            with Image.open(path) as image:
                image.verify()
            return
        payload = _download(record["image_url"])
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        # Commons에서 받은 JPEG 그대로 저장한다. 크롭·재인코딩하지 않는다.
        path.write_bytes(payload)
        print(f"Downloaded {record['id']}: {record['reference_mood']}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, records))
    (output / "samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def predict_samples(
    records: list[dict[str, str]], model_path: Path, device: str = "cpu", output: Path = OUTPUT,
) -> list[dict[str, str]]:
    """가중치를 한 번 읽고 모든 사진의 상위 3개 예측을 기록한다."""
    if device == "cpu":
        import torch
        torch.set_num_threads(2)
    network = moodfit.load_feature_extractor(model_path, device=device)
    results = []
    for record in records:
        prediction = moodfit.predict_mood(output / record["local_path"], network)
        top3 = sorted(prediction["mood_scores"].items(), key=lambda pair: pair[1], reverse=True)[:3]
        result = dict(record)
        result.update({
            "predicted_mood": prediction["mood"],
            "confidence": f"{prediction['confidence']:.6f}",
            "top3": " / ".join(f"{name} {score:.1%}" for name, score in top3),
            "model_path": str(network.checkpoint_path),
            "checkpoint_sha256": network.checkpoint_sha256,
        })
        results.append(result)
        print(f"{record['id']} [{record['reference_mood']}] -> {result['top3']}", flush=True)
    csv_path = output / "predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_gallery(results, output)
    return results


def write_gallery(results: list[dict[str, str]], output: Path = OUTPUT) -> None:
    cards = []
    for record in results:
        escape = html.escape
        cards.append(
            '<article>'
            f'<img src="{escape(record["local_path"], quote=True)}" alt="{escape(record["description"])}">'
            f'<h2>{escape(record["reference_mood"])} 참고 이미지</h2>'
            f'<p>모델 예측: <b>{escape(record["predicted_mood"])}</b> '
            f'({float(record["confidence"]):.1%})</p>'
            f'<p class="sub">TOP 3: {escape(record["top3"])}</p>'
            f'<p class="sub"><a href="{escape(record["source_url"], quote=True)}">원본·출처</a><br>'
            f'{escape(record["author"])} · {escape(record["license"])}</p></article>'
        )
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>MoodFit 외부 이미지 예측 테스트</title><style>
body{font-family:"Malgun Gothic",sans-serif;background:#f8fafc;color:#0f172a;max-width:1300px;margin:36px auto;padding:0 24px}
h1{font-size:28px}header p{line-height:1.7;color:#475569}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}
article{background:white;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;padding-bottom:14px}
article img{display:block;width:100%;height:320px;object-fit:contain;background:#eef2f6}article h2{font-size:17px}
article h2,article p{margin:12px 16px}.sub{font-size:12px;line-height:1.6;color:#64748b}a{color:#4f46e5}
</style><header><h1>인터넷 의상 사진 10개 · MoodFit 예측 결과</h1>
<p>참고 무드는 사진을 선정할 때 붙인 값이며, AI Hub의 공식 정답 라벨이 아닙니다.
이 10장은 외부 입력 동작 확인용입니다. 정식 정확도 평가는 별도의 정답 데이터로 진행해야 합니다.</p>
<p>모델 파일: MODEL_PATH</p></header><main class="grid">CARDS</main></html>'''
    document = document.replace("MODEL_PATH", html.escape(results[0]["model_path"]))
    document = document.replace("CARDS", "\n".join(cards))
    (output / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--refresh", nargs="*", default=[], metavar="ID", help="해당 번호의 사진을 다시 내려받기")
    args = parser.parse_args()
    if args.no_download:
        records = json.loads((OUTPUT / "samples.json").read_text(encoding="utf-8"))
    else:
        records = download_samples(refresh=set(args.refresh))
    if args.download_only:
        print(f"이미지와 출처 저장: {OUTPUT}")
        return
    model_path = args.model
    if model_path is None:
        candidates = [moodfit.DEFAULT_MODEL_PATH, ROOT / "Model" / "fashion_mood_model_pytorch.pth"]
        model_path = next((path for path in candidates if path.is_file()), candidates[0])
    predict_samples(records, model_path, args.device)
    print(f"사진·예측 결과 보기: {OUTPUT / 'report.html'}")


if __name__ == "__main__":
    main()
