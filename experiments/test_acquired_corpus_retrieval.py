"""Focused checks for the corpus retrieval experiment."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

SPEC = importlib.util.spec_from_file_location('corpus_retrieval', Path(__file__).with_name('04_acquired_corpus_retrieval.py'))
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class RetrievalTests(unittest.TestCase):
    def test_query_gallery_partition_is_stable_and_disjoint(self):
        rows = [{'section_id': str(i), 'paper_id': 'p'} for i in range(7)]
        queries, gallery, excluded = MODEL.partition(rows)
        self.assertEqual(MODEL.partition(list(reversed(rows))), (queries, gallery, excluded))
        self.assertFalse({x['section_id'] for x in queries} & {x['section_id'] for x in gallery})
        self.assertEqual(len(queries) + len(gallery), 7)
        self.assertEqual(MODEL.partition(rows[:3])[2], ['p'])

    def test_centroids_are_pooled_by_correct_paper_and_normalized(self):
        x = sparse.csr_matrix([[1., 0.], [1., 0.], [0., 1.]])
        rows = [{'paper_id': 'a'}, {'paper_id': 'a'}, {'paper_id': 'b'}]
        actual = MODEL.centroid_matrix(x, rows, ['b', 'a']).toarray()
        np.testing.assert_allclose(actual, [[0, 1], [1, 0]])

    def test_paper_macro_does_not_treat_paragraphs_as_independent_papers(self):
        queries = [dict(paper_id=p, section_id=str(i), source_locator='p') for i, p in enumerate(['a', 'a', 'a', 'b'])]
        result, _ = MODEL.evaluate(np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), queries, ['a', 'b'])
        self.assertEqual(result['micro']['recall_at_1'], .75)
        self.assertEqual(result['macro_over_papers']['recall_at_1'], .5)
        self.assertEqual(result['macro_over_papers']['mrr'], .75)

    def test_shared_text_cannot_be_both_query_and_gallery_under_different_papers(self):
        text = 'clinical evidence ' * 30
        rows = [dict(paper_id=p, section_id=p, section_name='Results', text=text, split=s)
                for p, s in [('a', 'train'), ('b', 'test')]]
        retained, reasons = MODEL.prepare(rows)
        self.assertEqual(retained, [])
        self.assertEqual(reasons['text_shared_between_papers'], 2)


if __name__ == '__main__':
    unittest.main()
