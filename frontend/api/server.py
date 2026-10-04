"""Local API that places new diseases with the v2 production model.

The frontend shows the prebuilt v2 graph as static files; placing a *new*
disease needs v2's fitted encoders and fusion, so it runs here.  The scoring
is v2's own: requests go through ``data_sources.record_from_user_input`` and
``place_disease.place`` unchanged.

Endpoints:
    GET  /api/health                      model status
    POST /api/place                       place one disease (body below)
    GET  /api/suggest?field=...&q=...     vocabulary lookup for the form
                                          (field: phenotypes, genes, drugs, ontology)
    GET  /api/patient/status              whether MedGemma is reachable
    POST /api/patient/interpret           {"text"}: patient's words -> HPO terms to confirm
    POST /api/patient/explain-disease     {"id", "name"}: Orphanet description in plain words
    POST /api/patient/explain-group       {"category", "examples", "shared"}: a group in plain words

POST /api/place body:
    {"disease": {<v2 JSON: name, description, phenotypes, genes, ...>},
     "id": "USER:...", "top": 20,
     "others": [{"id": "USER:...", "disease": {...}}, ...]}

``others`` are the user's other added diseases, so new diseases can also be
matched with each other.  Nothing is written to the v2 graph files.

When ``frontend/dist`` exists (``npm run build``), the app itself is served at
``/`` too, so ``python frontend/api/server.py`` is all a researcher needs.

Usage (from the repository root, after ``python v2/build_graph.py``):
    python frontend/api/server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import traceback
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient import MedGemma, PatientHelper  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = PROJECT_ROOT / "v2"
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
MAX_TOP = 50
MAX_BODY = 1_000_000
SUGGEST_FIELDS = ("phenotypes", "genes", "drugs", "ontology")


def support_level(neighbour: dict[str, Any]) -> str:
    """Same rule ``place_disease.add_to_graph`` uses for a new disease's edges."""

    if neighbour.get("known_relations"):
        return "curated"
    similarities = neighbour.get("similarities", {})
    return "plausible" if any(similarities.get(m, 0) > 0 for m in ("gene", "drug")) else "novel"


def to_json(value: Any) -> Any:
    """Make numpy scalars and arrays JSON serialisable."""

    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


class PlacementService:
    """Holds the v2 model and answers API requests.  ``loader`` returns
    ``(model, place, record_from_user_input)``; tests pass stubs."""

    def __init__(self, loader: Callable[[], tuple[Any, Callable, Callable]]):
        self._loader = loader
        self._lock = threading.Lock()
        self.status = "loading"
        self.error: Optional[str] = None
        self.model = None
        self._place = None
        self._parse = None
        self._index: dict[str, list[tuple[str, str, str]]] = {}

    def load(self) -> None:
        try:
            self.model, self._place, self._parse = self._loader()
            self._index = self._build_index()
            self.status = "ready"
        except Exception as error:  # Reported through /api/health.
            self.status = "error"
            self.error = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.error:
            out["error"] = self.error
        if self.model is not None:
            out["diseases"] = len(self.model.ids)
            out["modalities"] = list(self.model.engine.modalities)
        return out

    # -- placement ----------------------------------------------------------

    def place(self, body: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        document = body.get("disease")
        if not isinstance(document, dict) or not str(document.get("name", "")).strip():
            raise ValueError("Send {\"disease\": {...}} with at least a name.")
        top = max(1, min(int(body.get("top", 20)), MAX_TOP))
        query_id = str(body.get("id") or document.get("id") or "USER:new-disease")

        with self._lock:
            record, warnings = self._parse(document, self.model.knowledge)
            model = self.model
            others = [o for o in body.get("others") or [] if isinstance(o, dict) and o.get("id") != query_id]
            if others:
                # A shallow copy shares the fitted fusions and their caches;
                # extend() replaces (never mutates) the gallery arrays.
                model = copy.copy(self.model)
                ids, names, records = [], [], []
                for other in others:
                    other_record, _ = self._parse(other.get("disease") or {}, self.model.knowledge)
                    if other_record.get("name"):
                        ids.append(str(other["id"]))
                        names.append(other_record["name"])
                        records.append(other_record)
                model.extend(ids, names, records)
            results, scored = self._place(model, record, query_id, top)
            fusion = scored["fusion"]
            weights = {
                m: round(float(w), 4) for m, w in zip(fusion.modalities, fusion.similarity_weights) if float(w) > 0
            }
        for item in results:
            item["support"] = support_level(item)
        return to_json({
            "id": query_id,
            "name": record["name"],
            "present": scored["present"],
            "warnings": warnings,
            "weights": weights,
            "neighbours": results,
        })

    # -- suggestions --------------------------------------------------------

    def _build_index(self) -> dict[str, list[tuple[str, str, str]]]:
        k = self.model.knowledge
        hpo = []
        for term, label in k.hpo_labels.items():
            if k.is_phenotypic_abnormality(term):
                hpo.append((term, label, f"{term} {label}".lower()))
        drugs = [(chembl, label, f"{chembl} {label}".lower()) for chembl, label in k.drug_labels.items()]
        return {
            "phenotypes": sorted(hpo, key=lambda t: len(t[1])),
            "genes": sorted(((g, "", g.lower()) for g in k.gene_symbols), key=lambda t: (len(t[0]), t[0])),
            "drugs": sorted(drugs, key=lambda t: len(t[1])),
            "ontology": sorted(
                ((o, label, f"{o} {label}".lower()) for o, label in k.ontology_labels.items()), key=lambda t: len(t[1])
            ),
        }

    def suggest(self, field: str, query: str, limit: int = 10) -> list[dict[str, str]]:
        self._require_ready()
        if field not in SUGGEST_FIELDS:
            raise ValueError(f"field must be one of {', '.join(SUGGEST_FIELDS)}")
        q = query.strip().lower()
        if len(q) < 2:
            return []
        prefix, contains = [], []
        word = re.compile(rf"\b{re.escape(q)}")
        for term, label, key in self._index[field]:
            if key.startswith(q) or label.lower().startswith(q):
                prefix.append((term, label))
            elif word.search(key):
                contains.append((term, label))
            if len(prefix) >= limit:
                break
        return [{"id": t, "label": l} for t, l in (prefix + contains)[:limit]]

    def vocabulary(self, field: str) -> list[tuple[str, str]]:
        """(ID, label) pairs for a suggestion field; empty until the model has loaded."""

        return [(term, label) for term, label, _ in self._index.get(field, [])]

    def _require_ready(self) -> None:
        if self.status != "ready":
            raise RuntimeError(self.error or "The v2 model is still loading.")


class Descriptions:
    """Orphanet clinical descriptions from v2's cached bundle, loaded on first use."""

    def __init__(self, loader: Callable[[], dict[str, str]]):
        self._loader = loader
        self._texts: Optional[dict[str, str]] = None
        self._lock = threading.Lock()

    def __call__(self, disease_id: str) -> Optional[str]:
        with self._lock:
            if self._texts is None:
                self._texts = self._loader()
        return self._texts.get(disease_id) or None


def load_descriptions() -> dict[str, str]:
    sys.path.insert(0, str(V2_DIR))
    from data_sources import load_bundle

    return {d: r.get("description") or "" for d, r in load_bundle().records.items()}


def load_v2() -> tuple[Any, Callable, Callable]:
    """Load the production model the way ``place_disease.py`` does."""

    sys.path.insert(0, str(V2_DIR))
    from data_sources import record_from_user_input
    from place_disease import load_user_diseases, place
    from production import load_model

    model = load_model()
    users = load_user_diseases()  # Diseases added with place_disease.py --add.
    model.extend([u["id"] for u in users], [u["name"] for u in users], [u["record"] for u in users])
    model.pattern  # Build the submodel cache once, before request copies share it.
    return model, place, record_from_user_input


class Handler(SimpleHTTPRequestHandler):
    service: PlacementService

    def __init__(self, *args, service: PlacementService, patient: Optional[PatientHelper], directory: Optional[str], **kwargs):
        self.service = service
        self.patient = patient
        self.has_static = directory is not None
        super().__init__(*args, directory=directory or ".", **kwargs)

    def _send_json(self, status: int, document: Any) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _api(self, action: Callable[[], Any]) -> None:
        try:
            self._send_json(HTTPStatus.OK, action())
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/health":
            return self._api(self.service.health)
        if url.path == "/api/suggest":
            params = parse_qs(url.query)
            return self._api(lambda: self.service.suggest(params.get("field", [""])[0], params.get("q", [""])[0]))
        if url.path == "/api/patient/status":
            return self._api(lambda: self._patient().status())
        if url.path.startswith("/api/"):
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
        if not self.has_static:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Frontend not built; run `npm run build` in frontend/."})
        super().do_GET()

    def _patient(self) -> PatientHelper:
        if self.patient is None:
            raise RuntimeError("The patient view's language helper is not configured.")
        return self.patient

    def do_POST(self) -> None:
        routes: dict[str, Callable[[dict[str, Any]], Any]] = {
            "/api/place": self.service.place,
            "/api/patient/interpret": lambda b: self._patient().interpret(str(b.get("text", ""))),
            "/api/patient/explain-disease": lambda b: self._patient().explain_disease(str(b.get("id", "")), str(b.get("name", ""))),
            "/api/patient/explain-group": lambda b: self._patient().explain_group(b),
        }
        route = routes.get(urlparse(self.path).path)
        if route is None:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {error}"})
        if not isinstance(body, dict):
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Body must be a JSON object"})
        self._api(lambda: route(body))

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


def make_server(
    service: PlacementService,
    host: str,
    port: int,
    static_dir: Optional[Path],
    patient: Optional[PatientHelper] = None,
) -> ThreadingHTTPServer:
    directory = str(static_dir) if static_dir and static_dir.is_dir() else None
    return ThreadingHTTPServer((host, port), partial(Handler, service=service, patient=patient, directory=directory))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    service = PlacementService(load_v2)
    threading.Thread(target=service.load, daemon=True).start()
    patient = PatientHelper(
        MedGemma(),
        lambda field, q, limit: service.suggest(field, q, limit),
        Descriptions(load_descriptions),
        service.vocabulary,
    )
    server = make_server(service, args.host, args.port, DIST_DIR, patient)
    where = f"http://{args.host}:{args.port}"
    print(f"Loading the v2 model in the background; API at {where}/api/health")
    if DIST_DIR.is_dir():
        print(f"Serving the frontend at {where}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
