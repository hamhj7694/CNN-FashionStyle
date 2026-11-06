"""Exercise real viewer JavaScript with wheel, drag, reset and load events."""
import ast
import html
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ImageViewerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed for viewer JavaScript verification')
    def test_zoom_out_bounds_and_default_view(self):
        tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='image_viewer_html')
        ns={'html':html}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<viewer>','exec'),ns)
        documents=[ns['image_viewer_html']('https://example.test/look.webp','착장',kind) for kind in ('mf-selected-frame','mf-upload-frame')]
        result=subprocess.run([shutil.which('node'),str(ROOT/'tests/image_viewer_regression.cjs')],input=json.dumps(documents),capture_output=True,text=True,encoding='utf-8',timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
