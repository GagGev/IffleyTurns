"""Nine inductive, drug-free modality encoders used by the v4 embedding."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from data import Knowledge

MODALITY_DESCRIPTIONS = {
    "phenotype": "HPO phenotypes, ancestor-propagated and information-content weighted",
    "gene": "Curated disease genes",
    "pathway": "Reactome pathways of curated genes",
    "ontology": "Orphanet classification and Mondo ancestors",
    "name": "Disease name and synonyms (TF-IDF)",
    "text": "Clinical description (TF-IDF; own names, genes, and known drug words removed)",
    "inheritance": "Mode of inheritance",
    "onset": "Age of onset (ordinal kernel)",
    "prevalence": "Prevalence estimate (log-scale kernel)",
}
MODALITIES = tuple(MODALITY_DESCRIPTIONS)
TASK_MASKS = {
    "sibling": ("ontology", "name"),
    "gene": ("gene", "pathway"),
    "phenotype": ("phenotype",),
}

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
    name = ""

    def fit(self, train: Sequence[dict[str, Any]], catalogue: Sequence[dict[str, Any]]) -> "ModalityEncoder":
        raise NotImplementedError

    def transform(self, records: Sequence[dict[str, Any]]) -> sp.csr_matrix:
        raise NotImplementedError

    def feature_label(self, index: int) -> str:
        return str(index)


class WeightedItemEncoder(ModalityEncoder):
    def __init__(self, name: str, extract: Callable[[dict[str, Any]], dict[str, float]], label: Callable[[str], str]):
        self.name, self.extract, self.label = name, extract, label

    def fit(self, train, catalogue):
        items = sorted({item for record in catalogue for item in self.extract(record)})
        self.items_ = items
        self.index_ = {item: index for index, item in enumerate(items)}
        frequency: Counter[str] = Counter()
        annotated = 0
        for record in train:
            values = self.extract(record)
            if values:
                annotated += 1
                frequency.update(values)
        n = max(annotated, 1)
        self.idf_ = np.array([math.log((n + 1) / (frequency.get(item, 0) + 1)) for item in items], dtype=np.float32)
        return self

    def transform(self, records):
        rows, columns, values = [], [], []
        for row, record in enumerate(records):
            for item, weight in self.extract(record).items():
                column = self.index_.get(item)
                if column is not None and weight > 0:
                    rows.append(row)
                    columns.append(column)
                    values.append(weight * self.idf_[column])
        matrix = sp.csr_matrix(
            (values, (rows, columns)), shape=(len(records), len(self.items_)), dtype=np.float32
        )
        matrix.eliminate_zeros()
        return _normalize_rows(matrix)

    def feature_label(self, index):
        return self.label(self.items_[index])


class KernelEncoder(ModalityEncoder):
    def __init__(self, name: str, embed: Callable[[dict[str, Any]], Optional[np.ndarray]], labels: Sequence[str]):
        self.name, self.embed, self.labels = name, embed, list(labels)

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
    def __init__(self, name: str, document: Callable[[dict[str, Any]], str], ngram_range=(1, 2), min_df=2, max_df=0.2):
        self.name, self.document = name, document
        self.ngram_range, self.min_df, self.max_df = ngram_range, min_df, max_df

    def fit(self, train, catalogue):
        self.vectorizer_ = CountVectorizer(
            lowercase=True,
            stop_words=TEXT_STOP_WORDS,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        counts = self.vectorizer_.fit_transform([self.document(record) for record in train])
        nonempty = int((counts.getnnz(axis=1) > 0).sum())
        frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf_ = np.log((max(nonempty, 1) + 1) / (frequency + 1)).astype(np.float32) + 1.0
        self.terms_ = self.vectorizer_.get_feature_names_out()
        return self

    def transform(self, records):
        counts = self.vectorizer_.transform([self.document(record) for record in records]).astype(np.float32)
        counts.data = 1.0 + np.log(counts.data)
        return _normalize_rows(counts @ sp.diags(self.idf_))

    def feature_label(self, index):
        return str(self.terms_[index])


def _strip_gene_symbols(text: str, symbols: frozenset[str]) -> str:
    text = CYTOBAND.sub(" ", text)
    return TOKEN.sub(lambda match: " " if match.group(0) in symbols else match.group(0), text)


def _strip_words(text: str, words: frozenset[str]) -> str:
    return TOKEN.sub(lambda match: " " if match.group(0).lower() in words else match.group(0), text)


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
            vector += 1
            found = True
        elif value in ONSET_BINS:
            center = ONSET_BINS.index(value)
            vector += np.exp(-((positions - center) ** 2) / (2 * ONSET_SIGMA**2)).astype(np.float32)
            found = True
    return vector if found else None


def _prevalence_vector(record: dict[str, Any]) -> Optional[np.ndarray]:
    prevalence = record.get("prevalence")
    if prevalence is None or prevalence <= 0:
        return None
    value = float(np.clip(math.log10(prevalence), PREVALENCE_GRID[0], PREVALENCE_GRID[-1]))
    return np.exp(-((PREVALENCE_GRID - value) ** 2) / (2 * PREVALENCE_SIGMA**2)).astype(np.float32)


def _phenotype_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    items: dict[str, float] = {}
    for term, weight in record.get("phenotypes", {}).items():
        for ancestor in knowledge.hpo_ancestors(term):
            if ancestor != "HP:0000118" and weight > items.get(ancestor, 0):
                items[ancestor] = weight
    return items


def _pathway_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    return {pathway: 1.0 for gene in record.get("genes", {}) for pathway in knowledge.gene_pathways.get(gene, ())}


def _ontology_items(knowledge: Knowledge, record: dict[str, Any]) -> dict[str, float]:
    return {node: 1.0 for node in knowledge.expand_ontology(record.get("ontology_parents", ()))}


def _inheritance_items(record: dict[str, Any]) -> dict[str, float]:
    return {
        mapped: 1.0
        for value in record.get("inheritance", [])
        for mapped in INHERITANCE_MAP.get(str(value).strip().lower(), ())
    }


def _field_items(field: str, record: dict[str, Any]) -> dict[str, float]:
    return dict(record.get(field, {}))


def _description_document(knowledge: Knowledge, record: dict[str, Any]) -> str:
    text = _strip_own_names(record.get("description", "") or "", record)
    return _strip_words(_strip_gene_symbols(text, knowledge.gene_symbols), knowledge.drug_words)


def _name_document(knowledge: Knowledge, record: dict[str, Any]) -> str:
    return _strip_gene_symbols(" ; ".join([record.get("name", "")] + list(record.get("synonyms", []))), knowledge.gene_symbols)


def _identity(item: str) -> str:
    return item


def _label(mapping: str, knowledge: Knowledge, item: str) -> str:
    return f"{getattr(knowledge, mapping).get(item, item)} ({item})"


def build_encoders(knowledge: Knowledge) -> dict[str, ModalityEncoder]:
    label = lambda mapping: partial(_label, mapping, knowledge)
    return {
        "phenotype": WeightedItemEncoder("phenotype", partial(_phenotype_items, knowledge), label("hpo_labels")),
        "gene": WeightedItemEncoder("gene", partial(_field_items, "genes"), _identity),
        "pathway": WeightedItemEncoder("pathway", partial(_pathway_items, knowledge), label("pathway_labels")),
        "ontology": WeightedItemEncoder("ontology", partial(_ontology_items, knowledge), label("ontology_labels")),
        "name": TextEncoder("name", partial(_name_document, knowledge), ngram_range=(1, 2), max_df=0.05),
        "text": TextEncoder("text", partial(_description_document, knowledge), ngram_range=(1, 2), max_df=0.2),
        "inheritance": WeightedItemEncoder("inheritance", _inheritance_items, _identity),
        "onset": KernelEncoder("onset", _onset_vector, ONSET_BINS),
        "prevalence": KernelEncoder("prevalence", _prevalence_vector, [f"1e{value:+.2f}" for value in PREVALENCE_GRID]),
    }


class DiseaseEncoder:
    def __init__(self, knowledge: Knowledge, modalities: Iterable[str] = MODALITIES):
        all_encoders = build_encoders(knowledge)
        self.modalities = tuple(modalities)
        self.encoders = {name: all_encoders[name] for name in self.modalities}

    def fit(self, train: Sequence[dict[str, Any]], catalogue: Sequence[dict[str, Any]]) -> "DiseaseEncoder":
        for encoder in self.encoders.values():
            encoder.fit(train, catalogue)
        return self

    def transform(self, records: Sequence[dict[str, Any]]) -> dict[str, sp.csr_matrix]:
        return {name: encoder.transform(records) for name, encoder in self.encoders.items()}
