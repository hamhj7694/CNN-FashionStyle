"""Create resumable, resized WebP copies for the web without modifying originals."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def optimize_image(source: Path, asset_dir: Path, *, max_edge: int = 1280, quality: int = 80) -> dict:
    if max_edge < 1 or not 1 <= quality <= 100:
        raise ValueError('Invalid image size or WebP quality')
    stat = source.stat()
    signature = f'{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_edge}|{quality}|webp-v1'
    key = hashlib.sha256(signature.encode('utf-8')).hexdigest()[:24]
    target = asset_dir / f'full_{key}.webp'
    asset_dir.mkdir(parents=True, exist_ok=True)
    reused = target.is_file() and target.stat().st_size > 0
    if not reused:
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            temporary = target.with_suffix('.tmp')
            image.save(temporary, format='WEBP', quality=quality, method=3)
            os.replace(temporary, target)
    return dict(path=str(target.resolve()), key=f'full/{target.name}',
                original_bytes=stat.st_size, optimized_bytes=target.stat().st_size, reused=reused)


def prepare_catalog(metadata: Path, asset_dir: Path, output: Path, *, root: Path = ROOT,
                    max_edge: int = 1280, quality: int = 80, workers: int = 8,
                    limit: int | None = None, progress_path: Path | None = None) -> dict:
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('workers and limit must be positive')
    frame = pd.read_csv(metadata).fillna('')
    if 'original_img_path' not in frame:
        raise ValueError('metadata requires original_img_path')
    originals = frame.original_img_path.astype(str).drop_duplicates().tolist()
    if limit is not None:
        originals = originals[:limit]
        frame = frame[frame.original_img_path.isin(originals)].copy()
    started = time.monotonic()
    completed, failed = {}, {}
    original_bytes = optimized_bytes = reused = 0
    progress_path = progress_path or output.with_suffix('.progress.json')
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def report(finished=False):
        summary = dict(total_images=len(originals), completed_images=len(completed),
                       failed_images=len(failed), reused_images=reused,
                       original_bytes=original_bytes, optimized_bytes=optimized_bytes,
                       reduction_percent=round(100 * (1 - optimized_bytes / original_bytes), 2) if original_bytes else 0,
                       elapsed_seconds=round(time.monotonic() - started, 1), finished=finished,
                       output_catalog=str(output), max_edge=max_edge, quality=quality)
        temporary = progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, progress_path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary

    def run(value):
        source = Path(value)
        if not source.is_absolute():
            source = root / source
        return optimize_image(source, asset_dir, max_edge=max_edge, quality=quality)

    report()
    iterator = iter(originals)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        def submit():
            try:
                value = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(run, value)] = value
            return True
        for _ in range(workers * 3):
            if not submit():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                value = pending.pop(future)
                try:
                    result = future.result()
                    completed[value] = result
                    original_bytes += result['original_bytes']
                    optimized_bytes += result['optimized_bytes']
                    reused += int(result['reused'])
                except (OSError, ValueError) as error:
                    failed[value] = str(error)
                submit()
                if (len(completed) + len(failed)) % 1000 == 0:
                    report()
    valid = frame[frame.original_img_path.isin(completed)].copy()
    valid['full_image_path'] = valid.original_img_path.map(lambda value: completed[value]['path'])
    valid['full_image_key'] = valid.original_img_path.map(lambda value: completed[value]['key'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    valid.to_csv(temporary, index=False, encoding='utf-8-sig')
    os.replace(temporary, output)
    if failed:
        output.with_suffix('.errors.json').write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = report(finished=True)
    summary['catalog_rows'] = len(valid)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'recommend_pool/metadata.csv')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--asset-dir', type=Path, default=ROOT / 'recommend_pool/web_assets')
    parser.add_argument('--output', type=Path, default=ROOT / 'recommend_pool/metadata_optimized.csv')
    parser.add_argument('--max-edge', type=int, default=1280)
    parser.add_argument('--quality', type=int, default=80)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    prepare_catalog(args.metadata, args.asset_dir, args.output, root=args.root,
                    max_edge=args.max_edge, quality=args.quality, workers=args.workers, limit=args.limit)


if __name__ == '__main__':
    main()
