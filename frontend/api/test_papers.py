"""Tests for paper upload (papers.py) without loading MedGemma or the v2 model.

Run from the repository root:
    python -m unittest discover -s frontend/api
"""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from papers import PROJECT_ROOT, CompactClient, PaperPlacer, names_feature, title_of  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from v2_5.client import Completion, InvalidModelResponse  # noqa: E402

SCHEMA = {"properties": {"focal_disease": {"type": "string"}, "phenotypes": {"type": "array"}, "genes": {"type": "array"}}}


class StubInner:
    config = "config"

    def __init__(self, document):
        self.document = document
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, schema, prompt_version, use_cache=True):
        self.calls.append((system_prompt, prompt_version))
        return Completion(self.document, "medgemma", "", "sha", False, {}, "key")


def encode(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


class CompactClientTest(unittest.TestCase):
    def test_asks_for_short_output_under_its_own_cache_key(self):
        inner = StubInner({"focal_disease": "X", "phenotypes": []})
        CompactClient(inner).complete_json("System.", "User.", SCHEMA, "v1")
        system_prompt, version = inner.calls[0]
        self.assertTrue(system_prompt.startswith("System."))
        self.assertIn("at most 15 consecutive words", system_prompt)
        self.assertEqual(version, "v1+compact")

    def test_null_lists_become_empty_lists(self):
        inner = StubInner({"focal_disease": None, "phenotypes": None, "genes": ["FBN1"]})
        document = CompactClient(inner).complete_json("S", "U", SCHEMA, "v1").document
        self.assertEqual(document, {"focal_disease": None, "phenotypes": [], "genes": ["FBN1"]})

    def test_rejects_a_reply_that_is_not_the_extraction(self):
        # What parses out of a reply cut off at the token limit: one inner phenotype item.
        inner = StubInner({"label": "Seizure", "quote": "seizures"})
        with self.assertRaises(InvalidModelResponse):
            CompactClient(inner).complete_json("S", "U", SCHEMA, "v1")


class NamesFeatureTest(unittest.TestCase):
    def test_a_quote_must_contain_every_distinctive_word_of_the_label(self):
        quote = "Echocardiography confirmed a ventricular septal defect and a dilated aortic root."
        self.assertTrue(names_feature({"label": "Ventricular septal defect", "identifier": "HP:0001629", "quote": quote}))
        self.assertFalse(names_feature({"label": "Bicuspid aortic valve", "identifier": "HP:0001647", "quote": quote}))

    def test_genes_match_by_symbol_and_whole_word(self):
        quote = "Variants in TGFBR1 and SMAD3 cause the syndrome."
        self.assertTrue(names_feature({"feature_type": "gene", "label": "SMAD3", "identifier": "SMAD3", "quote": quote}))
        self.assertFalse(names_feature({"feature_type": "gene", "label": "TGFB", "identifier": "TGFB", "quote": quote}))


class PaperPlacerTest(unittest.TestCase):
    def setUp(self):
        self.placer = PaperPlacer(lambda: None, lambda record, query_id, top: [], base_url="http://127.0.0.1:9/v1")

    def assert_rejected(self, body, message):
        with self.assertRaises(ValueError) as caught:
            self.placer.place(body)
        self.assertIn(message, str(caught.exception))

    def test_rejects_unsupported_files(self):
        self.assert_rejected({"filename": "paper.docx", "content": encode("text")}, "Unsupported file type .docx")

    def test_rejects_content_that_is_not_base64(self):
        self.assert_rejected({"filename": "paper.txt", "content": "not base64!"}, "base64")

    def test_rejects_an_empty_file(self):
        self.assert_rejected({"filename": "paper.txt", "content": ""}, "empty")

    def test_rejects_a_file_with_no_text(self):
        self.assert_rejected({"filename": "paper.txt", "content": encode("   \n\n  ")}, "No text")

    def test_title_falls_back_to_the_first_line(self):
        units = [type("Unit", (), {"section": "body", "text": "Marfan syndrome in two siblings\nMore text"})()]
        self.assertEqual(title_of(units, "file"), "Marfan syndrome in two siblings")
        self.assertEqual(title_of([], "file"), "file")


if __name__ == "__main__":
    unittest.main()
