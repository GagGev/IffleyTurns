"""Inductive per-modality disease encoders.

Every encoder turns a disease record into a non-negative, L2-normalized
sparse vector, so the dot product of two rows is a cosine similarity in
[0, 1].  An all-zero row means the modality is unavailable for that disease.

Item vocabularies are built from the full catalogue (they only define column
positions), while all learned statistics -- IDF / information-content weights
and text IDF -- are fitted on training diseases only.  A disease that was not
seen during fitting, including a user-supplied new disease, is therefore
encoded exactly like a held-out test disease.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from data_sources import Knowledge

# Modalities and the database families they come from.  Benchmarks mask the
# modalities derived from their own label source.
MODALITY_DESCRIPTIONS = {
    "phenotype": "HPO phenotypes (Orphanet product 4, Open Targets/HPO), ancestor-propagated, information-content weighted",
    "gene": "Curated genes (Orphanet product 6, HPO, ClinGen, ClinVar)",
    "pathway": "Reactome pathways of curated genes (Open Targets target annotations)",
    "ot_gene": "Open Targets direct gene associations with overall score >= 0.5",
    "drug": "Drugs in clinical trials or approved (Open Targets) and FDA/EMA orphan designations",
    "drug_target": "Molecular targets of those drugs (Open Targets / ChEMBL mechanisms)",
    "ontology": "Orphanet classification and Mondo ancestors",
    "name": "Disease name and synonyms (TF-IDF)",
    "text": "Orphanet clinical description (TF-IDF, own names and gene symbols removed)",
    "inheritance": "Mode of inheritance (Orphanet product 9)",
    "onset": "Age of onset (Orphanet product 9), ordinal kernel",
    "prevalence": "Prevalence estimate (Orphanet product 9), log-scale kernel",
}
MODALITIES = tuple(MODALITY_DESCRIPTIONS)

INHERITANCE_MAP = {
    "autosomal recessive": ("AR",),
    "autosomal dominant": ("AD",),
    "semidominant": ("AD", "semidominant"),
    "semi-dominant": ("AD", "semidominant"),
    "not applicable": ("not genetic",),
    "x-linked recessive": ("X-linked", "XLR"),
    "x-linked dominant": ("X-linked", "XLD"),
    "x-linked": ("X-linked",),
    "multigenic/multifactorial": ("multifactorial",),
    "oligogenic": ("oligogenic", "multifactorial"),
    "mitochondrial inheritance": ("mitochondrial",),
    "mitochondrial": ("mitochondrial",),
    "y-linked": ("Y-linked",),
}
ONSET_BINS = ("Antenatal", "Neonatal", "Infancy", "Childhood", "Adolescent", "Adult", "Elderly")
ONSET_SIGMA = 0.75
PREVALENCE_GRID = np.arange(-8.0, -2.24, 0.25)
PREVALENCE_SIGMA = 0.5
CYTOBAND = re.compile(r"\b(?:\d{1,2}|X|Y)[pq]\d+(?:\.\d+)?(?:-[pq]?\d+(?:\.\d+)?)?\b")
TOKEN = re.compile(r"[A-Za-z0-9]+")
TEXT_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"rare", "disease", "disorder", "syndrome", "characterized", "patients"})


def _normalize_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / norms) @ matrix, dtype=np.float32)


class ModalityEncoder:
    name: str = ""

    def fit(self, train: Sequence[dict[str, Any]], catalogue: Sequence[dict[str, Any]]) -> "ModalityEncoder":
        raise NotImplementedError

    def transform(self, records: Sequence[dict[str, Any]]) -> sp.csr_matrix:
        raise NotImplementedError

    def feature_label(self, index: int) -> str:
        return str(index)


class WeightedItemEncoder(ModalityEncoder):
    """Weighted item sets with IDF (information content) learned on training diseases."""

    def __init__(self, name: str, extract: Callable[[dict[str, Any]], dict[str, float]], label: Callable[[str], str]):
        self.name = name
        self.extract = extract
        self.label = label

    def fit(self, train, catalogue):
        items: set[str] = set()
        for record in catalogue:
            items.update(self.extract(record))
        for record in train:
            items.update(self.extract(record))
        self.items_ = sorted(items)
        self.index_ = {item: i for i, item in enumerate(self.items_)}
        frequency = Counter()
        annotated = 0
        for record in train:
            extracted = self.extract(record)
            if extracted:
                annotated += 1
                frequency.update(extracted.keys())
        n = max(annotated, 1)
        self.idf_ = np.array(
            [math.log((n + 1) / (frequency.get(item, 0) + 1)) for item in self.items_], dtype=np.float32
        )
        self.n_train_annotated_ = annotated
        return self

    def transform(self, records):
        rows, cols, values = [], [], []
        for row, record in enumerate(records):
            for item, weight in self.extract(record).items():
                column = self.index_.get(item)
                if column is not None and weight > 0:
                    rows.append(row)
                    cols.append(column)
                    values.append(weight * self.idf_[column])
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(records), len(self.items_)), dtype=np.float32)
        matrix.eliminate_zeros()
        return _normalize_rows(matrix)

    def feature_label(self, index):
        item = self.items_[index]
        return self.label(item)


class KernelEncoder(ModalityEncoder):
    """Fixed (unlearned) kernel embeddings for ordinal or numeric attributes."""

    def __init__(self, name: str, embed: Callable[[dict[str, Any]], Optional[np.ndarray]], labels: Sequence[str]):
        self.name = name
        self.embed = embed
        self.labels = list(labels)

    def fit(self, train, catalogue):
        return self

    def transform(self, records):
        dense = np.zeros((len(records), len(self.labels)), dtype=np.float32)
        for row, record in enumerate(records):
            vector = self.embed(record)
            if vector is not None:
                dense[row] = vector
        return _normalize_rows(sp.csr_matrix(dense))

    def feature_label(self, index):
        return self.labels[index]


class TextEncoder(ModalityEncoder):
    """Sublinear TF-IDF with vocabulary and IDF from training diseases only.

    The document-frequency filter must not count held-out diseases: a term
    shared by a new disease and a single catalogue disease is dropped in
    deployment, so it must also be dropped when a test disease is the query.
    """

    def __init__(self, name: str, document: Callable[[dict[str, Any]], str], ngram_range=(1, 2), min_df=2, max_df=0.2):
        self.name = name
        self.document = document
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, train, catalogue):
        self.vectorizer_ = CountVectorizer(
            lowercase=True,
            stop_words=TEXT_STOP_WORDS,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        counts = self.vectorizer_.fit_transform([self.document(r) for r in train])
        nonempty = int((counts.getnnz(axis=1) > 0).sum())
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf_ = np.log((max(nonempty, 1) + 1) / (df + 1)).astype(np.float32) + 1.0
        self.terms_ = self.vectorizer_.get_feature_names_out()
        return self

    def transform(self, records):
        counts = self.vectorizer_.transform([self.document(r) for r in records]).astype(np.float32)
        counts.data = 1.0 + np.log(counts.data)
        return _normalize_rows(counts @ sp.diags(self.idf_))

    def feature_label(self, index):
        return str(self.terms_[index])


def _strip_gene_symbols(text: str, gene_symbols: frozenset[str]) -> str:
    text = CYTOBAND.sub(" ", text)
    return TOKEN.sub(lambda m: " " if m.group(0) in gene_symbols else m.group(0), text)


def _strip_own_names(text: str, record: dict[str, Any]) -> str:
    names = sorted({record.get("name", "")} | set(record.get("synonyms", [])), key=len, reverse=True)
    for name in names:
        if name and len(name) >= 3:
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def _onset_vector(record: dict[str, Any]) -> Optional[np.ndarray]:
    vector = np.zeros(len(ONSET_BINS), dtype=np.float32)
    positions = np.arange(len(ONSET_BINS))
    found = False
    for value in record.get("onset", []):
        if value == "All ages":
            vector += 1.0
            found = True
        elif value in ONSET_BINS:
            center = ONSET_BINS.index(value)
            vector += np.exp(-((positions - center) ** 2) / (2 * ONSET_SIGMA**2)).astype(np.float32)
            found = True
    return vector if found else None


def _prevalence_vector(record: dict[str, Any]) -> Optional[np.ndarray]:
    prevalence = record.get("prevalence")
    if prevalence is None or not prevalence > 0:
        return None
    value = float(np.clip(math.log10(prevalence), PREVALENCE_GRID[0], PREVALENCE_GRID[-1]))
    return np.exp(-((PREVALENCE_GRID - value) ** 2) / (2 * PREVALENCE_SIGMA**2)).astype(np.float32)


# Module-level extractors (bound with functools.partial) keep encoders picklable.


def _phenotype_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    items: dict[str, float] = {}
    for term, weight in record.get("phenotypes", {}).items():
        for ancestor in knowledge.hpo_ancestors(term):
            if ancestor != "HP:0000118" and weight > items.get(ancestor, 0.0):
                items[ancestor] = weight
    return items


def _pathway_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    items = dict(record.get("pathways", {}))
    for gene, gene_weight in record.get("genes", {}).items():
        for pathway in knowledge.gene_pathways.get(gene, ()):
            if gene_weight > items.get(pathway, 0.0):
                items[pathway] = gene_weight
    return items


def _drug_target_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    items: dict[str, float] = {}
    for drug, weight in record.get("drugs", {}).items():
        for target in knowledge.drug_targets.get(drug, ()):
            if weight > items.get(target, 0.0):
                items[target] = weight
    return items


def _ontology_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    return {node: 1.0 for node in knowledge.expand_ontology(record.get("ontology_parents", ()))}


def _inheritance_items(record: dict[str, Any]) -> dict[str, float]:
    return {
        mapped: 1.0
        for value in record.get("inheritance", [])
        for mapped in INHERITANCE_MAP.get(str(value).strip().lower(), ())
    }


def _field_items(field_name: str, record: dict[str, Any]) -> dict[str, float]:
    return dict(record.get(field_name, {}))


def _description_document(knowledge: Knowledge, record: dict[str, Any]) -> str:
    text = _strip_own_names(record.get("description", "") or "", record)
    return _strip_gene_symbols(text, knowledge.gene_symbols)


def _name_document(knowledge: Knowledge, record: dict[str, Any]) -> str:
    text = " ; ".join([record.get("name", "")] + list(record.get("synonyms", [])))
    return _strip_gene_symbols(text, knowledge.gene_symbols)


def _identity_label(item: str) -> str:
    return item


def _mapped_label(mapping_name: str, knowledge: Knowledge, item: str) -> str:
    return f"{getattr(knowledge, mapping_name).get(item, item)} ({item})"


def build_encoders(knowledge: Knowledge) -> dict[str, ModalityEncoder]:
    """Create one unfitted encoder per modality."""

    def label(mapping_name: str) -> Callable[[str], str]:
        return partial(_mapped_label, mapping_name, knowledge)

    return {
        "phenotype": WeightedItemEncoder("phenotype", partial(_phenotype_items, knowledge), label("hpo_labels")),
        "gene": WeightedItemEncoder("gene", partial(_field_items, "genes"), _identity_label),
        "pathway": WeightedItemEncoder("pathway", partial(_pathway_items, knowledge), label("pathway_labels")),
        "ot_gene": WeightedItemEncoder("ot_gene", partial(_field_items, "ot_genes"), _identity_label),
        "drug": WeightedItemEncoder("drug", partial(_field_items, "drugs"), label("drug_labels")),
        "drug_target": WeightedItemEncoder("drug_target", partial(_drug_target_items, knowledge), _identity_label),
        "ontology": WeightedItemEncoder("ontology", partial(_ontology_items, knowledge), label("ontology_labels")),
        "name": TextEncoder("name", partial(_name_document, knowledge), ngram_range=(1, 2), min_df=2, max_df=0.05),
        "text": TextEncoder("text", partial(_description_document, knowledge), ngram_range=(1, 2), min_df=2, max_df=0.2),
        "inheritance": WeightedItemEncoder("inheritance", _inheritance_items, _identity_label),
        "onset": KernelEncoder("onset", _onset_vector, ONSET_BINS),
        "prevalence": KernelEncoder(
            "prevalence", _prevalence_vector, [f"1e{v:+.2f} per person" for v in PREVALENCE_GRID]
        ),
    }


class DiseaseEncoder:
    """Fit and apply all modality encoders together."""

    def __init__(self, knowledge: Knowledge, modalities: Iterable[str] = MODALITIES):
        all_encoders = build_encoders(knowledge)
        self.modalities = tuple(modalities)
        self.encoders = {m: all_encoders[m] for m in self.modalities}

    def fit(self, train: Sequence[dict[str, Any]], catalogue: Sequence[dict[str, Any]]) -> "DiseaseEncoder":
        for encoder in self.encoders.values():
            encoder.fit(train, catalogue)
        return self

    def transform(self, records: Sequence[dict[str, Any]]) -> dict[str, sp.csr_matrix]:
        return {m: encoder.transform(records) for m, encoder in self.encoders.items()}

    def shared_features(
        self, modality: str, row_a: sp.csr_matrix, row_b: sp.csr_matrix, top: int = 5
    ) -> list[tuple[str, float]]:
        """Return the features contributing most to a modality's cosine similarity."""

        product = row_a.multiply(row_b).tocoo()
        if product.nnz == 0:
            return []
        order = np.argsort(-product.data)[:top]
        encoder = self.encoders[modality]
        return [(encoder.feature_label(int(product.col[i])), float(product.data[i])) for i in order]
