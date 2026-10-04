"""Tests for the patient view's MedGemma helpers, with a stub in place of the model.

    python3 -m unittest frontend/api/test_patient.py -v
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402
from patient import MedGemma, MedGemmaUnavailable, PatientHelper, flips_meaning, parse_terms, stem, word_match  # noqa: E402
from test_server import StubModel, stub_parse, stub_place  # noqa: E402

VOCABULARY = [
    {"id": "HP:0001250", "label": "Seizure"},
    {"id": "HP:0001252", "label": "Hypotonia"},
    {"id": "HP:0012638", "label": "Abnormal nervous system physiology including seizure"},
]


def suggest(field: str, query: str, limit: int) -> list[dict[str, str]]:
    q = query.lower()
    return [v for v in VOCABULARY if q in v["label"].lower()][:limit]


class StubMedGemma:
    def __init__(self, reply: str = '```json\n{"terms": ["Seizures", "Hypotonia", "Glowing toes"]}\n```'):
        self.reply = reply
        self.calls = 0

    def available(self) -> bool:
        return True

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        self.calls += 1
        return self.reply


class PatientHelperTest(unittest.TestCase):
    def test_parse_terms_handles_code_fences_and_prose(self):
        self.assertEqual(parse_terms('```json\n{"terms": ["Seizure", " Ataxia "]}\n```'), ["Seizure", "Ataxia"])
        self.assertEqual(parse_terms("- Seizure\n- Ataxia"), ["Seizure", "Ataxia"])

    def test_interpret_maps_terms_to_hpo_with_exact_match_first(self):
        helper = PatientHelper(StubMedGemma(), suggest, lambda _: None)
        terms = helper.interpret("my son has fits and is floppy")["terms"]
        self.assertEqual([t["term"] for t in terms], ["Seizures", "Hypotonia", "Glowing toes"])
        # "Seizures" has no label containing it; the singular is tried and the exact label ranks first.
        self.assertEqual(terms[0]["matches"][0], {"id": "HP:0001250", "label": "Seizure"})
        self.assertEqual(terms[1]["matches"], [{"id": "HP:0001252", "label": "Hypotonia"}])
        self.assertEqual(terms[2]["matches"], [], "unknown terms are returned unmatched for the patient to search")

    def test_interpret_validates_and_caches(self):
        model = StubMedGemma()
        helper = PatientHelper(model, suggest, lambda _: None)
        with self.assertRaises(ValueError):
            helper.interpret("   ")
        helper.interpret("fits")
        helper.interpret("fits")
        # One call to read the terms, one for alternative names of the unmatched one; both cached.
        self.assertEqual(model.calls, 2)

    def test_word_match_ignores_order_and_endings(self):
        vocabulary = [("HP:0025336", "Delayed ability to sit"), ("HP:0001263", "Global developmental delay")]
        self.assertEqual(stem("sitting"), "sit")
        self.assertEqual(word_match("Delayed sitting", vocabulary, 3), [{"id": "HP:0025336", "label": "Delayed ability to sit"}])
        self.assertEqual(word_match("of the", vocabulary, 3), [])
        aorta = [("HP:0004942", "Aortic aneurysm"), ("HP:0030962", "Abnormal aortic morphology")]
        self.assertEqual(word_match("Abnormality of the aorta", aorta, 3)[0]["label"], "Abnormal aortic morphology")

    def test_matches_never_reverse_the_meaning(self):
        self.assertTrue(flips_meaning("Skin hyperelasticity", "Lack of skin elasticity"))
        self.assertTrue(flips_meaning("Skin elasticity", "Decreased skin elasticity"))
        self.assertFalse(flips_meaning("Reduced muscle tone", "Decreased muscle tone".replace("Decreased", "Reduced")))
        self.assertFalse(flips_meaning("Hypotonia", "Hypotonia"))

    def test_unmatched_terms_get_alternative_names(self):
        class Model(StubMedGemma):
            def chat(self, system, user, max_tokens=300):
                self.calls += 1
                if "official Human Phenotype Ontology" in system:
                    return '{"Fits": ["Seizure"]}'
                return '{"terms": ["Fits"]}'

        helper = PatientHelper(Model(), suggest, lambda _: None)
        self.assertEqual(helper.interpret("fits")["terms"][0]["matches"][0]["label"], "Seizure")

    def test_explain_disease_needs_a_source_description(self):
        helper = PatientHelper(StubMedGemma("Plain words."), suggest, {"ORPHA:558": "Clinical text."}.get)
        self.assertEqual(helper.explain_disease("ORPHA:558", "Marfan syndrome")["text"], "Plain words.")
        self.assertEqual(helper.explain_disease("ORPHA:1", "Unknown"), {"id": "ORPHA:1", "text": None})
        with self.assertRaises(ValueError):
            helper.explain_disease("", "")

    def test_medgemma_must_be_local(self):
        with self.assertRaises(ValueError):
            MedGemma("https://api.example.com/v1")
        MedGemma("http://localhost:9/v1")  # allowed

    def test_unreachable_medgemma_raises_unavailable(self):
        client = MedGemma("http://127.0.0.1:9/v1", timeout=1)
        self.assertFalse(client.available())
        with self.assertRaises(MedGemmaUnavailable):
            client.chat("system", "user")


class PatientApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        service = server.PlacementService(lambda: (StubModel(), stub_place, stub_parse))
        service.load()
        helper = PatientHelper(StubMedGemma(), suggest, {"ORPHA:1": "Clinical text."}.get)
        cls.httpd = server.make_server(service, "127.0.0.1", 0, None, helper)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_routes(self):
        with urllib.request.urlopen(self.base + "/api/patient/status") as response:
            self.assertEqual(json.loads(response.read()), {"medgemma": "ready"})
        status, body = self.post("/api/patient/interpret", {"text": "fits"})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["terms"]), 3)
        status, body = self.post("/api/patient/explain-disease", {"id": "ORPHA:1", "name": "Alpha"})
        self.assertEqual((status, set(body)), (200, {"id", "text"}))
        status, _ = self.post("/api/patient/explain-group", {"category": "Rare neurologic disease", "examples": ["Alpha"]})
        self.assertEqual(status, 200)
        status, body = self.post("/api/patient/interpret", {"text": ""})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
