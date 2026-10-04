from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from modalities import MODALITIES
from model import EmbeddingModel, EmbeddingNet, split_pairs, to_torch


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
