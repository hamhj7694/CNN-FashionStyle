from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

from prepare_web_images import optimize_image, prepare_catalog
from upload_web_images import upload_catalog
from r2_storage import upload_file_to_r2


class WebAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='moodfit-web-test-')
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.jpg'
        Image.effect_noise((1600, 2400), 40).convert('RGB').save(self.source, quality=95)
        self.before = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_resize_compression_original_preserved_and_resume(self):
        result = optimize_image(self.source, self.root / 'assets')
        with Image.open(result['path']) as image:
            self.assertEqual(image.format, 'WEBP')
            self.assertEqual(image.size, (853, 1280))
        self.assertLess(result['optimized_bytes'], result['original_bytes'])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.before)
        self.assertTrue(optimize_image(self.source, self.root / 'assets')['reused'])
        small = self.root / 'small.png'
        Image.new('RGB', (60, 100), 'green').save(small)
        tiny = optimize_image(small, self.root / 'assets')
        with Image.open(tiny['path']) as image:
            self.assertEqual(image.size, (60, 100))

    def test_deduplicated_catalog_missing_source_and_resume(self):
        metadata = self.root / 'metadata.csv'
        pd.DataFrame([
            dict(id='a', original_img_path='source.jpg'),
            dict(id='b', original_img_path='source.jpg'),
            dict(id='missing', original_img_path='missing.jpg'),
        ]).to_csv(metadata, index=False)
        output = self.root / 'prepared.csv'
        args = dict(metadata=metadata, asset_dir=self.root/'assets', output=output, root=self.root, workers=2)
        summary = prepare_catalog(**args)
        self.assertEqual((summary['completed_images'], summary['failed_images'], summary['catalog_rows']), (1,1,2))
        rows = pd.read_csv(output)
        self.assertEqual(rows.full_image_key.nunique(),1)
        self.assertEqual(prepare_catalog(**args)['reused_images'],1)
        self.assertTrue(output.with_suffix('.errors.json').exists())

    def prepared_csv(self):
        asset = self.root / 'asset.webp'
        Image.new('RGB',(80,120),'blue').save(asset,format='WEBP')
        csv = self.root / 'prepared.csv'
        pd.DataFrame([dict(id=key, original_img_path='dataset/a.jpg',full_image_path=str(asset),full_image_key='full/a.webp') for key in ('a','b')]).to_csv(csv,index=False)
        return csv

    def test_upload_dry_run_and_url_only_public_catalog(self):
        prepared = self.prepared_csv()
        target = self.root / 'metadata_r2.csv'
        with patch('upload_web_images.get_r2_settings') as settings:
            self.assertEqual(upload_catalog(prepared,target,dry_run=True)['images'],1)
            settings.assert_not_called()
            self.assertFalse(target.exists())
        config=dict(account_id='test-account',bucket_name='test-bucket',public_base_url='https://assets.example.test')
        with patch('upload_web_images.get_r2_settings',return_value=config), patch('upload_web_images.create_r2_client',return_value=object()), patch('upload_web_images.upload_file_to_r2',return_value='https://assets.example.test/full/a.webp') as upload:
            upload_catalog(prepared,target,workers=2)
            self.assertEqual(upload.call_count,1)
            upload_catalog(prepared,target,workers=2)
            self.assertEqual(upload.call_count,1)
        result=pd.read_csv(target)
        self.assertEqual(len(result),2)
        self.assertNotIn('full_image_path',result)
        self.assertNotIn('full_image_key',result)
        self.assertTrue(result.full_image_url.eq('https://assets.example.test/full/a.webp').all())

    def test_webp_mime_overrides_windows_registry(self):
        self.prepared_csv()
        client = Mock()
        settings = dict(bucket_name='test-bucket', public_base_url='https://assets.example.test')
        with patch('r2_storage.mimetypes.guess_type', return_value=('application/octet-stream', None)):
            url = upload_file_to_r2(self.root/'asset.webp', 'full/a.webp', client, settings)
        self.assertEqual(url, 'https://assets.example.test/full/a.webp')
        headers = client.upload_file.call_args.kwargs['ExtraArgs']
        self.assertEqual(headers['ContentType'], 'image/webp')
        self.assertEqual(headers['CacheControl'], 'public, max-age=31536000, immutable')

    def test_failed_upload_does_not_replace_catalog(self):
        prepared=self.prepared_csv()
        output=self.root/'metadata_r2.csv'
        output.write_text('existing catalog',encoding='utf-8')
        config=dict(account_id='test-account',bucket_name='test-bucket',public_base_url='https://assets.example.test')
        with patch('upload_web_images.get_r2_settings',return_value=config), patch('upload_web_images.create_r2_client',return_value=object()), patch('upload_web_images.upload_file_to_r2',side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):
                upload_catalog(prepared,output)
        self.assertEqual(output.read_text(encoding='utf-8'),'existing catalog')


class DeploymentEntryTests(unittest.TestCase):
    def test_brand_styles_survive_streamlit_rerun(self):
        from streamlit.testing.v1 import AppTest
        entry = Path(__file__).resolve().parents[1] / 'deploy/streamlit_app.py'
        page = AppTest.from_file(str(entry), default_timeout=60).run()
        self.assertFalse(page.exception)
        self.assertTrue(any('.mf-nav { display:flex' in block.value for block in page.markdown))
        page.run()
        self.assertFalse(page.exception)
        self.assertTrue(any('.mf-nav { display:flex' in block.value for block in page.markdown))


if __name__ == '__main__':
    unittest.main()
