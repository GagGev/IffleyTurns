"""Focused tests for the resumable relationship acquisition pipeline."""
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "acquire_relationships", Path(__file__).with_name("acquire_relationships.py")
)
acquire = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acquire
SPEC.loader.exec_module(acquire)


class FakeResponse:
    def __init__(self, body=b'{"ok": true}', status=200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class RelationshipAcquisitionTests(unittest.TestCase):
    def test_deduplication_uses_identifiers_then_normalized_title(self):
        with tempfile.TemporaryDirectory() as directory:
            db = acquire.connect(Path(directory) / "papers.sqlite")
            try:
                first = acquire.upsert_paper(
                    db,
                    {
                        "pmid": "123",
                        "doi": "https://doi.org/10.1/ABC",
                        "title": "The Rare-Disease Comparison",
                    },
                    "Europe PMC",
                )
                second = acquire.upsert_paper(
                    db,
                    {
                        "pmid": "123",
                        "title": "Updated metadata title",
                        "abstract": "Abstract.",
                    },
                    "PubMed",
                )
                third = acquire.upsert_paper(
                    db,
                    {"title": "Rare Disease Comparison"},
                    "PubMed",
                )
                db.commit()
                self.assertEqual(first, second)
                self.assertEqual(first, third)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
                sources = json.loads(
                    db.execute("SELECT sources_json FROM papers").fetchone()[0]
                )
                self.assertEqual(sources, ["Europe PMC", "PubMed"])
            finally:
                db.close()

    def test_orpha_mapping_flags_ambiguous_aliases_without_guessing(self):
        mapper = acquire.DiseaseMapper(
            [
                ("alpha syndrome", ["ORPHA:1"]),
                ("beta syndrome", ["ORPHA:2", "ORPHA:3"]),
            ]
        )
        mentions = mapper.detect(
            "Alpha syndrome can resemble beta-syndrome in early childhood."
        )
        self.assertEqual(mentions[0].orpha_id, "ORPHA:1")
        self.assertEqual(mentions[0].mapping_status, "exact")
        self.assertIsNone(mentions[1].orpha_id)
        self.assertEqual(mentions[1].mapping_status, "ambiguous")
        self.assertEqual(mentions[1].candidate_ids, ("ORPHA:2", "ORPHA:3"))

    def test_scoring_validation_preserves_zero_and_na_semantics(self):
        score, rule, _ = acquire.score_passage(
            "The disorders have no shared clinical phenotype."
        )
        self.assertEqual((score, rule), (0, "explicit_non_overlap"))
        score, rule, _ = acquire.score_passage(
            "Both disorders are caused by the same causal gene."
        )
        self.assertEqual((score, rule), (4, "very_strong_equivalence"))
        acquire.validate_score(None, 1)
        acquire.validate_score(0, 1)
        with self.assertRaises(ValueError):
            acquire.validate_score(5, 1)
        with self.assertRaises(ValueError):
            acquire.validate_score(None, 0)

    def test_http_success_is_cached_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            db = acquire.connect(data_dir / "papers.sqlite")
            client = acquire.ApiClient(db, data_dir)
            try:
                with mock.patch.object(client, "_delay"), mock.patch(
                    "urllib.request.urlopen", return_value=FakeResponse()
                ) as urlopen:
                    first, _ = client.get(
                        "Europe PMC",
                        "https://example.test/api",
                        {"q": "rare"},
                        "search",
                    )
                    second, _ = client.get(
                        "Europe PMC",
                        "https://example.test/api",
                        {"q": "rare"},
                        "search",
                    )
                self.assertEqual(first, second)
                urlopen.assert_called_once()
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM search_log").fetchone()[0], 1
                )
            finally:
                db.close()

    def test_repeated_evaluation_is_idempotent_and_abstract_is_low_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            db = acquire.connect(Path(directory) / "papers.sqlite")
            mapper = acquire.DiseaseMapper(
                [
                    ("alpha syndrome", ["ORPHA:1"]),
                    ("beta syndrome", ["ORPHA:2"]),
                ]
            )
            try:
                db.executemany(
                    """INSERT INTO orpha_entities(
                    orpha_id,preferred_name,entity_type,source_sha256
                    ) VALUES(?,?,?,?)""",
                    [
                        ("ORPHA:1", "Alpha syndrome", "Disease", "test"),
                        ("ORPHA:2", "Beta syndrome", "Disease", "test"),
                    ],
                )
                acquire.upsert_paper(
                    db,
                    {
                        "pmid": "1",
                        "title": "Comparison of two rare disorders",
                        "abstract": (
                            "Alpha syndrome and Beta syndrome share clinical symptoms "
                            "and have similar progression."
                        ),
                    },
                    "PubMed",
                )
                db.commit()
                acquire.evaluate_papers(db, mapper)
                first_counts = (
                    db.execute("SELECT COUNT(*) FROM paper_pairs").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM dimension_scores").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0],
                )
                acquire.evaluate_papers(db, mapper)
                second_counts = (
                    db.execute("SELECT COUNT(*) FROM paper_pairs").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM dimension_scores").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0],
                )
                self.assertEqual(first_counts, (1, 5, 5))
                self.assertEqual(first_counts, second_counts)
                confidences = {
                    row[0]
                    for row in db.execute(
                        """SELECT extraction_confidence FROM dimension_scores
                        WHERE ordinal_score IS NOT NULL"""
                    )
                }
                self.assertEqual(confidences, {1})
                self.assertEqual(
                    db.execute(
                        """SELECT COUNT(*) FROM dimension_scores
                        WHERE score_label='NA' AND ordinal_score IS NULL"""
                    ).fetchone()[0],
                    3,
                )
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
