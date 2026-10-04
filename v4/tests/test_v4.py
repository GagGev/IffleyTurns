from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from fusion import CosineEngine, LinearFusion, combine_scores
from modalities import MODALITIES, TASK_MASKS
from model import (
    EmbeddingModel,
    EmbeddingNet,
    best_retrieval_epoch,
    phenotype_pairs,
    retrieval_metrics,
    sample_negative_matrix,
    sampled_softmax_loss,
    split_pairs,
    to_torch,
)


def tiny_inputs(n: int = 5):
    rng = np.random.default_rng(2)
    matrices = {
        name: sp.csr_matrix(rng.random((n, 3), dtype=np.float32))
        for name in MODALITIES
    }
    target = torch.device("cpu")
    return matrices, {name: to_torch(matrix, target) for name, matrix in matrices.items()}


def test_embedding_is_normalized_and_cosine_is_symmetric():
    matrices, inputs = tiny_inputs()
    config = {"projection_dim": 4, "embedding_dim": 6, "dropout": 0.0}
    net = EmbeddingNet({name: matrix.shape[1] for name, matrix in matrices.items()}, config).eval()
    available = torch.ones((5, len(MODALITIES)), dtype=torch.bool)
    embedding, _ = net.encode(inputs, available, torch.zeros(len(MODALITIES), dtype=torch.bool))
    np.testing.assert_allclose(torch.linalg.vector_norm(embedding, dim=1).detach(), 1, atol=1e-5)
    scores = (embedding @ embedding.T).detach().numpy()
    np.testing.assert_allclose(scores, scores.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(scores), 1, atol=1e-5)


def test_masked_modalities_receive_zero_attention():
    matrices, inputs = tiny_inputs()
    config = {"projection_dim": 4, "embedding_dim": 6, "dropout": 0.0}
    net = EmbeddingNet({name: matrix.shape[1] for name, matrix in matrices.items()}, config).eval()
    available = torch.ones((5, len(MODALITIES)), dtype=torch.bool)
    masked = torch.tensor([name in ("ontology", "name") for name in MODALITIES])
    _, attention = net.encode(inputs, available, masked)
    for name in ("ontology", "name"):
        assert torch.count_nonzero(attention[:, MODALITIES.index(name)]) == 0
    assert TASK_MASKS["phenotype"] == ("phenotype",)


def test_training_pairs_cannot_contain_validation_or_test_diseases():
    pairs = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
    splits = np.array(["train", "train", "validation", "test"])
    result = split_pairs(pairs, splits)
    assert result["train"].tolist() == [[0, 1]]
    assert result["validation"].tolist() == [[0, 2]]
    assert result["test"].tolist() == [[1, 3], [2, 3]]
    assert all(splits[index] == "train" for pair in result["train"] for index in pair)


def test_score_queries_preserves_requested_rows_and_catalogue_columns():
    embedding = np.array([[1, 0], [0, 1], [2**-0.5, 2**-0.5]], dtype=np.float32)
    model = EmbeddingModel(
        ids=["a", "b", "c"],
        names=["A", "B", "C"],
        encoder=None,
        matrices={},
        available=np.empty((3, 0), dtype=bool),
        state={},
        config={},
        embeddings_full=embedding,
        history=[],
        diagnostics={},
    )
    scores = model.score_queries(["c", "a"])
    assert scores.shape == (2, 3)
    np.testing.assert_allclose(scores[0], embedding[2] @ embedding.T)
    np.testing.assert_allclose(scores[1], embedding[0] @ embedding.T)


def test_sampled_softmax_rewards_positive_above_negatives():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    good = sampled_softmax_loss(
        embeddings, np.array([0]), np.array([1]), np.array([[2]]), 0.1
    )
    bad = sampled_softmax_loss(
        embeddings, np.array([0]), np.array([2]), np.array([[1]]), 0.1
    )
    assert good < bad


def test_negative_sampling_excludes_union_of_known_positives():
    known = [{1, 2}, {0}, {0}, set()]
    negatives = sample_negative_matrix(
        np.array([0, 0]), 20, np.arange(4), known, np.random.default_rng(0)
    )
    assert set(negatives.ravel()) == {3}


def test_phenotype_pairs_and_retrieval_metrics():
    phenotypes = sp.csr_matrix(
        np.array(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    )
    pairs = phenotype_pairs(phenotypes, neighbours=1, min_similarity=0.5)
    assert pairs.tolist() == [[0, 1]]
    embeddings = torch.nn.functional.normalize(torch.from_numpy(phenotypes.toarray()), dim=1)
    metrics = retrieval_metrics(embeddings, pairs, np.array([0, 1]), np.arange(3))
    assert metrics["map"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["hits@10"] == 1.0


def test_cosine_engine_hides_masked_modalities():
    matrices = {name: sp.csr_matrix(np.eye(3, dtype=np.float32)) for name in MODALITIES}
    available = np.ones((3, len(MODALITIES)), dtype=bool)
    engine = CosineEngine(matrices, available)
    S, A = engine.pairs(np.array([0]), np.array([1]), masked=("ontology", "name"))
    assert S[0, MODALITIES.index("ontology")] == 0
    assert not A[0, MODALITIES.index("name")]


def test_hybrid_mix_zero_matches_embedding():
    embedding = np.array([[1.0, 0.2], [0.0, 3.0]], dtype=np.float32)
    fusion = np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
    mixed = combine_scores(embedding, fusion, 1.0, 1.0, 0.0)
    np.testing.assert_allclose(mixed, embedding)


def test_score_queries_adds_fusion_when_mix_is_one():
    matrices = {name: sp.csr_matrix(np.eye(2, dtype=np.float32)) for name in MODALITIES}
    available = np.ones((2, len(MODALITIES)), dtype=bool)
    fusion = LinearFusion(
        modalities=tuple(MODALITIES),
        similarity_weights=np.ones(len(MODALITIES), dtype=np.float32),
        availability_weights=np.zeros(len(MODALITIES), dtype=np.float32),
        intercept=0.0,
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = EmbeddingModel(
        ids=["a", "b"],
        names=["A", "B"],
        encoder=None,
        matrices=matrices,
        available=available,
        state={},
        config={},
        embeddings_full=embeddings,
        history=[],
        diagnostics={},
        fusion=fusion,
        embedding_scale=1.0,
        fusion_scale=1.0,
        fusion_mix=1.0,
    )
    scores = model.score_queries(["a"])
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


def test_checkpoint_selection_uses_map_not_auc():
    history = [
        {"validation_mean_map": 0.2, "validation_mean_auc": 0.9},
        {"validation_mean_map": 0.4, "validation_mean_auc": 0.8},
    ]
    assert best_retrieval_epoch(history) == 1
