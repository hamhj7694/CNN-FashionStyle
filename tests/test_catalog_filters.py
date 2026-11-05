"""Catalog filter conditions, similarity row alignment and real Streamlit state transitions."""
from __future__ import annotations

import ast
import itertools
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from curation import filter_catalog, get_recommendation_sections

ROOT = Path(__file__).resolve().parents[1]


class CatalogRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            dict(id=str(i), mood=mood, category=category, gender=gender)
            for i, (mood, category, gender) in enumerate(itertools.product(
                ['클래식', '모던'], ['상의', '하의'], ['남성', '여성', '공용']))
        ])

    def test_all_mood_category_combinations_and_all_genders(self):
        original = self.catalog.copy(deep=True)
        for mood, category in itertools.product(['전체', '클래식', '모던'], ['전체', '상의', '하의']):
            with self.subTest(mood=mood, category=category):
                expected = self.catalog
                if mood != '전체':
                    expected = expected[expected.mood == mood]
                if category != '전체':
                    expected = expected[expected.category == category]
                actual = filter_catalog(self.catalog, '전체', category, mood)
                pd.testing.assert_frame_equal(actual, expected)
                sections = get_recommendation_sections(None, category, self.catalog, selected_mood=mood)
                rows = sections['similar'] + sections['discovery']
                self.assertTrue(all(row['id'] in set(expected.id) for row in rows))
                self.assertEqual(len(rows), len({row['id'] for row in rows}))
        pd.testing.assert_frame_equal(self.catalog, original)

    def test_similarity_positions_with_duplicate_ids_and_nonunique_indexes(self):
        catalog = pd.DataFrame([
            dict(id='a', mood='모던', category='상의'),
            dict(id='a', mood='모던', category='상의'),
            dict(id='b', mood='모던', category='상의'),
            dict(id='c', mood='클래식', category='하의'),
        ], index=[7, 7, 99, 3])
        matrix = np.array([[1., 0.], [0., 1.], [.8, .6], [0., 1.]])
        sections = get_recommendation_sections([0., 1.], '상의', catalog, matrix, selected_mood='모던')
        self.assertEqual([row['id'] for row in sections['similar']], ['b', 'a'])
        self.assertAlmostEqual(sections['similar'][0]['similarity'], .6)
        self.assertEqual(sections['discovery'], [])

    def test_empty_and_incompatible_embeddings(self):
        self.assertEqual(get_recommendation_sections(None, '신발', self.catalog), {'similar': [], 'discovery': []})
        self.assertEqual(get_recommendation_sections(None, '전체', self.catalog.iloc[:0]), {'similar': [], 'discovery': []})
        result = get_recommendation_sections([1., 0.], '상의', self.catalog, np.ones((2, 3)), selected_mood='클래식')
        self.assertTrue(result['similar'])
        self.assertTrue(all(row['category'] == '상의' and row['mood'] == '클래식' for row in result['similar']))
        self.assertFalse(any('similarity' in row for row in result['similar']))


class CatalogFilterUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'app.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        guard = next(node for node in tree.body if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))
        source = '\n'.join(source.splitlines()[:guard.lineno - 1])
        source = source.replace('from __future__ import annotations', '', 1)
        prefix = f'from __future__ import annotations\nimport sys\nsys.path.insert(0, {str(ROOT)!r})\n'
        overrides = textwrap.dedent('''
            from types import SimpleNamespace
            HAS_SHADCN = False
            fixture_model = SimpleNamespace(checkpoint_sha256='fixture')
            fixture_records = [
                dict(id=str(i), mood=mood, category=category, gender='공용')
                for i, (mood, category) in enumerate(
                    (m, c) for m in ('클래식', '모던', '로맨틱') for c in ('상의', '하의') for _ in range(6)
                )
            ] + [dict(id='shoes', mood='클래식', category='신발', gender='공용')]
            # Include whitespace and null values to exercise option normalization.
            fixture_records[0]['mood'] = ' 클래식 '
            fixture_records[0]['category'] = ' 상의 '
            fixture_catalog = _normalize_metadata(pd.DataFrame(fixture_records))
            if st.session_state.get('fixture_mode') == 'empty':
                fixture_catalog = fixture_catalog.iloc[:0]
            load_metadata = lambda *args: (fixture_catalog, False)
            load_model_safely = lambda *args: (fixture_model, '')
            st.session_state['test_cards'] = {}
            st.session_state['test_embedding_calls'] = 0
            def fixture_file(*args, **kwargs):
                st.markdown('fixture-upload-control')
                mode = st.session_state.get('fixture_mode', 'none')
                if mode in ('none', 'empty'):
                    return None
                payload = io.BytesIO()
                Image.new('RGB', (80, 160), 'red' if mode == 'alternate' else 'blue').save(payload, format='PNG')
                payload.seek(0)
                return payload
            st.file_uploader = fixture_file
            def fixture_analysis(upload, model):
                mood = '없는무드' if st.session_state.get('fixture_mode') == 'unknown' else '클래식'
                return dict(mood=mood, confidence=.7, mood_scores={mood:.7,'모던':.2,'로맨틱':.1}, embedding=np.ones(256)), ''
            analyze_upload = fixture_analysis
            def fixture_embeddings(*args):
                st.session_state['test_embedding_calls'] += 1
                return None, ''
            load_item_embeddings = fixture_embeddings
            def record_card(item, section, index):
                st.session_state['test_cards'].setdefault(section, []).append(item)
                st.markdown(f"{section}: {item['id']}")
            render_card = record_card
            # Keep image rendering out of filter tests (covered by viewer/selection tests).
            render_interactive_image = lambda *args, **kwargs: None
            render_selected_recommendation = lambda *args, **kwargs: None
            main()
        ''')
        cls.script = prefix + source + '\n' + overrides

    def setUp(self):
        self.at = AppTest.from_string(self.script).run(timeout=30)
        self.assert_ok()

    def assert_ok(self):
        self.assertEqual(len(self.at.exception), 0, [e.message for e in self.at.exception])

    def cards(self):
        sections = self.at.session_state['test_cards']
        return sections.get('similar', []) + sections.get('discovery', [])

    def select(self, key, value):
        self.at.selectbox(key=key).select(value).run(timeout=30)
        self.assert_ok()

    def test_visible_labels_no_gender_and_clean_options(self):
        self.assertEqual(len(self.at.toggle), 0)
        self.assertEqual([box.label for box in self.at.selectbox], ['무드', '의류 분류'])
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')
        self.assertEqual(self.at.selectbox(key='category_filter').value, '전체')
        self.assertEqual(self.at.selectbox(key='mood_filter').options, ['전체', '로맨틱', '모던', '클래식'])
        self.assertTrue(any('37개' in item.value for item in self.at.caption))

    def test_both_filters_are_strict_and_manual_changes_persist(self):
        self.select('mood_filter', '모던')
        self.select('category_filter', '상의')
        self.assertEqual(len(self.cards()), 6)
        self.assertTrue(all(row['mood'] == '모던' and row['category'] == '상의' for row in self.cards()))
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '모던')
        self.assertEqual(self.at.selectbox(key='category_filter').value, '상의')

    def test_zero_results_and_reset(self):
        self.select('mood_filter', '모던')
        self.select('category_filter', '신발')
        self.assertEqual(self.cards(), [])
        self.assertEqual(self.at.session_state['test_embedding_calls'], 0)
        self.assertTrue(any('0개' in item.value for item in self.at.caption))
        self.assertTrue(any('조건에 맞는 아이템이 없습니다' in item.value for item in self.at.info))
        self.at.button(key='reset_catalog_filters').click().run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')
        self.assertEqual(self.at.selectbox(key='category_filter').value, '전체')
        self.assertTrue(self.cards())

    def test_upload_defaults_strict_filter_and_reset_while_uploaded(self):
        self.at.session_state['fixture_mode'] = 'upload'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '클래식')
        self.assertTrue(all(row['mood'] == '클래식' for row in self.cards()))
        self.select('mood_filter', '모던')
        self.assertTrue(all(row['mood'] == '모던' for row in self.cards()))
        self.select('category_filter', '하의')
        self.assertTrue(all(row['mood'] == '모던' and row['category'] == '하의' for row in self.cards()))
        self.at.button(key='reset_catalog_filters').click().run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')
        self.assertEqual(self.at.selectbox(key='category_filter').value, '전체')

    def test_new_upload_and_removed_upload(self):
        self.at.session_state['fixture_mode'] = 'upload'
        self.at.run(timeout=30)
        self.select('mood_filter', '모던')
        self.at.session_state['fixture_mode'] = 'alternate'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '클래식')
        self.at.session_state['fixture_mode'] = 'none'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')

    def test_unknown_options_and_missing_diagnosed_mood(self):
        # Seed stale session values before widget creation. AppTest cannot serialize
        # an invalid value into an already-created selectbox before running the app.
        self.at = AppTest.from_string(self.script)
        self.at.session_state['mood_filter'] = '삭제된 무드'
        self.at.session_state['category_filter'] = '삭제된 분류'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')

        self.assertEqual(self.at.selectbox(key='category_filter').value, '전체')
        self.at.session_state['fixture_mode'] = 'unknown'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertEqual(self.at.selectbox(key='mood_filter').value, '전체')

    def test_empty_catalog(self):
        self.at.session_state['fixture_mode'] = 'empty'
        self.at.run(timeout=30)
        self.assert_ok()
        self.assertTrue(all(box.disabled for box in self.at.selectbox))
        self.assertEqual(self.cards(), [])
        self.assertTrue(any('추천 데이터가 아직 없습니다' in item.value for item in self.at.info))


if __name__ == '__main__':
    unittest.main()
