"""Tests for the placement API's HTTP layer, with a stub in place of the v2 model.

The scoring itself is v2's (tested in v2/tests); these tests check request
validation, the response shape the frontend relies on, the support rule, and
that "others" are placed on a copy without touching the shared model.

    python3 -m unittest frontend/api/test_server.py -v
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402

MODALITIES = ("phenotype", "gene", "drug")


class StubModel:
    def __init__(self):
        self.ids = ["ORPHA:1", "ORPHA:2"]
        self.names = ["Alpha syndrome", "Beta disease"]
        self.engine = SimpleNamespace(modalities=MODALITIES)
        self.knowledge = SimpleNamespace(
            hpo_labels={"HP:0001250": "Seizure", "HP:0001263": "Global developmental delay", "HP:0000118": "Phenotypic abnormality"},
            is_phenotypic_abnormality=lambda term: term != "HP:0000118",
            gene_symbols=frozenset({"CDKL5", "FBN1"}),
            drug_labels={"CHEMBL191": "LOSARTAN"},
            ontology_labels={"ORPHA:102369": "Rare epilepsy"},
        )

    def extend(self, ids, names, records):
        self.ids = list(self.ids) + list(ids)
        self.names = list(self.names) + list(names)


def stub_parse(document, knowledge):
    warnings = [f"Ignored phenotype {p!r}" for p in document.get("phenotypes", []) if p not in knowledge.hpo_labels]
    return {"name": str(document.get("name", "")).strip()}, warnings


def stub_place(model, record, query_id, top):
    fusion = SimpleNamespace(modalities=MODALITIES, similarity_weights=[0.9, 0.5, 0.0])
    results = [
        {"rank": i + 1, "id": d, "name": n, "score": 1.0 - i, "percentile": 0.99,
         "similarities": {"gene": 0.4} if i == 0 else {"phenotype": 0.2},
         "known_relations": {}, "explanation": [], "contributions": {}, "annotation_adjustment": 0.0}
        for i, (d, n) in enumerate(zip(model.ids, model.names))
    ][:top]
    return results, {"fusion": fusion, "present": ["phenotype"]}


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = StubModel()
        cls.service = server.PlacementService(lambda: (cls.model, stub_place, stub_parse))
        cls.service.load()
        cls.httpd = server.make_server(cls.service, "127.0.0.1", 0, None)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def request(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        status, body = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["modalities"], list(MODALITIES))

    def test_place_returns_neighbours_with_support_and_weights(self):
        status, body = self.request("/api/place", {"disease": {"name": "New", "phenotypes": ["HP:0001250", "bad"]}, "id": "USER:new"})
        self.assertEqual(status, 200)
        self.assertEqual(body["id"], "USER:new")
        self.assertEqual(body["warnings"], ["Ignored phenotype 'bad'"])
        self.assertEqual(body["weights"], {"phenotype": 0.9, "gene": 0.5})
        self.assertEqual([n["support"] for n in body["neighbours"]], ["plausible", "novel"])

    def test_place_requires_a_name(self):
        status, body = self.request("/api/place", {"disease": {"phenotypes": []}})
        self.assertEqual(status, 400)
        self.assertIn("name", body["error"])

    def test_others_are_candidates_without_changing_the_shared_model(self):
        status, body = self.request("/api/place", {
            "disease": {"name": "New"}, "id": "USER:new",
            "others": [{"id": "USER:other", "disease": {"name": "Other"}}, {"id": "USER:new", "disease": {"name": "Self"}}],
        })
        self.assertEqual(status, 200)
        self.assertEqual([n["id"] for n in body["neighbours"]], ["ORPHA:1", "ORPHA:2", "USER:other"])
        self.assertEqual(self.model.ids, ["ORPHA:1", "ORPHA:2"])

    def test_suggest(self):
        status, body = self.request("/api/suggest?field=phenotypes&q=sei")
        self.assertEqual(body, [{"id": "HP:0001250", "label": "Seizure"}])
        _, body = self.request("/api/suggest?field=phenotypes&q=phenotypic")
        self.assertEqual(body, [], "terms outside 'Phenotypic abnormality' are not offered")
        _, body = self.request("/api/suggest?field=drugs&q=losar")
        self.assertEqual(body, [{"id": "CHEMBL191", "label": "LOSARTAN"}])
        status, _ = self.request("/api/suggest?field=nonsense&q=abc")
        self.assertEqual(status, 400)

    def test_invalid_json_and_unknown_endpoint(self):
        req = urllib.request.Request(self.base + "/api/place", data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        self.assertEqual(caught.exception.code, 400)
        status, _ = self.request("/api/nope")
        self.assertEqual(status, 404)

    def test_loading_failure_is_reported(self):
        def fail():
            raise FileNotFoundError("production_model.joblib not found")

        service = server.PlacementService(fail)
        service.load()
        self.assertEqual(service.health()["status"], "error")
        with self.assertRaises(RuntimeError):
            service.place({"disease": {"name": "x"}})


if __name__ == "__main__":
    unittest.main()
