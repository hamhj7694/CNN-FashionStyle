"""Upload optimized WebP files and publish a URL-only R2 catalog when complete."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from r2_storage import create_r2_client, get_r2_settings, upload_file_to_r2

ROOT = Path(__file__).resolve().parent


def upload_catalog(metadata: Path, output: Path, *, workers: int = 8, dry_run: bool = False) -> dict:
    if workers < 1:
        raise ValueError('workers must be positive')
    frame = pd.read_csv(metadata).fillna('')
    required = {'full_image_path', 'full_image_key'}
    if required.difference(frame.columns):
        raise ValueError('Run prepare_web_images.py first')
    assets = frame[['full_image_path', 'full_image_key']].drop_duplicates('full_image_key')
    if assets.empty:
        raise ValueError('No prepared images')
    missing = [value for value in assets.full_image_path if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} prepared images are missing')
    total_bytes = sum(Path(value).stat().st_size for value in assets.full_image_path)
    summary = dict(images=len(assets), catalog_rows=len(frame), bytes=total_bytes, dry_run=dry_run)
    if dry_run:
        print(json.dumps(summary), flush=True)
        return summary
    settings = get_r2_settings()
    client = create_r2_client(settings)
    state_path = output.with_suffix('.uploads.json')
    previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    destination = settings['public_base_url'].rstrip('/')
    state_key = f"{settings['account_id']}/{settings['bucket_name']}/{destination}"
    uploaded = previous.get('objects', {}) if previous.get('destination') == state_key else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(assets.itertuples(index=False, name=None))

    def upload(row):
        path, key = row
        if key in uploaded:
            return key, uploaded[key]
        return key, upload_file_to_r2(path, key, client, settings)

    def checkpoint():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(destination=state_key, objects=uploaded)), encoding='utf-8')
        os.replace(temporary, state_path)

    executor = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    iterator = iter(rows)
    count = 0
    def submit():
        try:
            row = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(upload, row))
        return True
    try:
        for _ in range(workers * 2):
            if not submit():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                key, url = future.result()
                uploaded[key] = url
                count += 1
                if count % 100 == 0:
                    checkpoint()
                    print(f'Uploaded {count}/{len(rows)}', flush=True)
                submit()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        checkpoint()
    public = frame.drop(columns=['full_image_path', 'crop_image_path'], errors='ignore').copy()
    public['full_image_url'] = public.full_image_key.map(uploaded)
    if public.full_image_url.isna().any():
        raise RuntimeError('Incomplete upload: active catalog was not published')
    public = public.drop(columns=['full_image_key'])
    temporary = output.with_suffix('.tmp')
    public.to_csv(temporary, index=False, encoding='utf-8-sig')
    os.replace(temporary, output)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'recommend_pool/metadata_optimized.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'recommend_pool/metadata_r2.csv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    upload_catalog(args.metadata, args.output, workers=args.workers, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
