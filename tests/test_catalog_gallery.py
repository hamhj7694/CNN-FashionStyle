"""Full catalog pagination and navigation retain filters and upload state."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from curation import filter_catalog
import test_catalog_filters as fixtures

ROOT = Path(__file__).resolve().parents[1]


class CatalogPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'catalog_image_page')
        ns = {'pd': pd, 'filter_catalog': filter_catalog}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<catalog-page>', 'exec'), ns)
        cls.page = staticmethod(ns['catalog_image_page'])

    def test_all_photos_reachable_without_page_overlap(self):
        data = pd.DataFrame([dict(id=str(i), mood='모던', category='상의', full_image_url=f'https://example.test/{i}.webp') for i in range(137)])
        pages = [self.page(data, '전체', '전체', n)[0] for n in (1, 2, 3)]
        self.assertEqual([len(p) for p in pages], [60, 60, 17])
        ids = [item for p in pages for item in p.id]
        self.assertEqual(len(set(ids)), 137)
        self.assertEqual(set(ids), set(data.id))
        self.assertEqual(self.page(data, '전체', '전체', 0)[2], 1)
        self.assertEqual(self.page(data, '전체', '전체', 999)[2], 3)
        self.assertEqual(self.page(data, '전체', '전체', 1)[0].id.tolist(), pages[0].id.tolist())

    def test_filter_before_photo_dedup_and_local_path_fallback(self):
        data = pd.DataFrame([
            dict(id='a', mood='모던', category='상의', full_image_url=' https://example.test/a.webp ', original_img_path=''),
            dict(id='b', mood='모던', category='하의', full_image_url='https://example.test/a.webp', original_img_path=''),
            dict(id='c', mood='클래식', category='상의', full_image_url=None, original_img_path='local/a.jpg'),
            dict(id='d', mood='클래식', category='상의', full_image_url='', original_img_path='local/a.jpg'),
        ])
        self.assertEqual(self.page(data, '전체', '전체')[1], 2)
        rows, total, _ = self.page(data, '모던', '하의')
        self.assertEqual(total, 1)
        self.assertEqual(rows.id.tolist(), ['b'])
        self.assertEqual(self.page(data, '모던', '원피스')[1], 0)

    def test_empty_and_invalid_page_size(self):
        rows, total, page = self.page(pd.DataFrame(columns=['id', 'mood', 'category']), '전체', '전체', 99)
        self.assertTrue(rows.empty)
        self.assertEqual((total, page), (0, 1))
        with self.assertRaises(ValueError):
            self.page(pd.DataFrame(), '전체', '전체', page_size=0)


class CatalogGalleryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CatalogFilterUITests.setUpClass()
        cls.script = fixtures.CatalogFilterUITests.script.replace(
            'fixture_catalog = _normalize_metadata(pd.DataFrame(fixture_records))',
            "fixture_records += [dict(id='extra_'+str(i), mood='모던', category='상의', gender='공용') for i in range(130)]\nfixture_catalog = _normalize_metadata(pd.DataFrame(fixture_records))",
            1,
        )

    def setUp(self):
        self.at = AppTest.from_string(self.script).run(timeout=30)
        self.ok()

    def ok(self):
        self.assertFalse(self.at.exception, [e.message for e in self.at.exception])

    def test_more_preserves_filters_and_back_navigation(self):
        self.at.selectbox(key='mood_filter').select('모던').run()
        self.at.selectbox(key='category_filter').select('상의').run()
        self.at.button(key='more_similar').click().run()
        self.ok()
        self.assertTrue(self.at.session_state['show_catalog'])
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '모던')
        self.assertEqual(self.at.selectbox(key='category_filter').value, '상의')
        self.assertEqual(sum(b.key.startswith('select_catalog_') for b in self.at.button), 60)
        self.at.button(key='close_catalog').click().run()
        self.ok()
        self.assertFalse(self.at.session_state['show_catalog'])
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '모던')

    def test_page_change_filter_change_and_empty_reset(self):
        self.at.button(key='more_catalog').click().run()
        self.at.button(key='catalog_next').click().run()
        self.ok()
        self.assertEqual(self.at.number_input(key='catalog_page').value, 2)
        self.at.selectbox(key='mood_filter').select('클래식').run()
        self.ok()
        self.assertEqual(self.at.number_input(key='catalog_page').value, 1)
        self.assertTrue(self.at.button(key='catalog_next').disabled)
        self.at.selectbox(key='category_filter').select('신발').run()
        self.at.selectbox(key='mood_filter').select('모던').run()
        self.ok()
        self.assertFalse(any(b.key.startswith('select_catalog_') for b in self.at.button))
        self.assertTrue(self.at.session_state['show_catalog'])
        self.at.button(key='reset_catalog_filters').click().run()
        self.ok()
        self.assertEqual(self.at.number_input(key='catalog_page').value, 1)

    def test_upload_and_selection_survive_gallery_round_trip(self):
        self.at.session_state['fixture_mode'] = 'upload'
        self.at.run()
        signature = self.at.session_state['upload_signature']
        self.at.button(key='more_top_mood').click().run()
        self.ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '클래식')
        self.at.button(key='select_catalog_0').click().run()
        self.ok()
        self.assertTrue(self.at.session_state['show_catalog'])
        self.assertIsNotNone(self.at.session_state['catalog_selected_item'])
        self.assertEqual(self.at.session_state['upload_signature'], signature)
        self.at.button(key='close_catalog').click().run()
        self.ok()
        self.assertFalse(self.at.session_state['show_catalog'])
        self.assertEqual(self.at.session_state['upload_signature'], signature)
        self.assertIsNotNone(self.at.session_state['selected_item'])

    def test_select_switch_and_close_preview_stay_in_gallery(self):
        self.at.button(key='more_catalog').click().run()
        self.at.button(key='select_catalog_0').click().run()
        self.ok()
        first_id = self.at.session_state['catalog_selected_item']['id']
        self.assertTrue(self.at.session_state['show_catalog'])
        self.at.button(key='select_catalog_1').click().run()
        self.ok()
        self.assertNotEqual(self.at.session_state['catalog_selected_item']['id'], first_id)
        self.assertEqual(self.at.number_input(key='catalog_page').value, 1)
        self.at.button(key='close_catalog_preview').click().run()
        self.ok()
        self.assertIsNone(self.at.session_state['catalog_selected_item'])
        self.assertTrue(self.at.session_state['show_catalog'])
        self.assertEqual(sum(b.key.startswith('select_catalog_') for b in self.at.button), 60)

    def test_preview_survives_page_change_and_clears_on_filter_change(self):
        self.at.button(key='more_catalog').click().run()
        self.at.button(key='select_catalog_0').click().run()
        selected_id = self.at.session_state['catalog_selected_item']['id']
        self.at.button(key='catalog_next').click().run()
        self.ok()
        self.assertEqual(self.at.number_input(key='catalog_page').value, 2)
        self.assertEqual(self.at.session_state['catalog_selected_item']['id'], selected_id)
        self.at.button(key='select_catalog_60').click().run()
        self.ok()
        self.assertNotEqual(self.at.session_state['catalog_selected_item']['id'], selected_id)
        self.at.selectbox(key='mood_filter').select('클래식').run()
        self.ok()
        self.assertIsNone(self.at.session_state['catalog_selected_item'])
        self.assertEqual(self.at.number_input(key='catalog_page').value, 1)
        self.assertTrue(self.at.session_state['show_catalog'])

    def test_page_input_clamps_when_result_count_changes(self):
        self.at.button(key='more_catalog').click().run()
        self.at.number_input(key='catalog_page').set_value(3).run()
        self.ok()
        self.assertTrue(self.at.button(key='catalog_next').disabled)
        self.at.selectbox(key='mood_filter').select('클래식').run()
        self.ok()
        self.assertEqual(self.at.number_input(key='catalog_page').value, 1)


if __name__ == '__main__':
    unittest.main()
