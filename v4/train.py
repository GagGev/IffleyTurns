"""Train the one v4 embedding model.

Usage:
    python v4/train.py
    python v4/train.py --epochs 40 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import EVALUATION_DIR, MODEL_DIR, configure_stdout, timed, write_json
from data import load_bundle
from modalities import MODALITIES, DiseaseEncoder
from model import (
    DEFAULT_CONFIG,
    EmbeddingModel,
    build_relation_pairs,
    encode_full,
    train_network,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
    configure_stdout()

    config = {
        **DEFAULT_CONFIG,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    bundle = load_bundle(use_cache=not args.rebuild_data)
    ids = sorted(bundle.records)
    records = [bundle.records[disease] for disease in ids]
    train_records = [
        bundle.records[disease]
        for disease in ids
        if bundle.splits[disease] == "train"
    ]
    with timed(f"Fitting nine encoders on {len(train_records):,} train diseases"):
        encoder = DiseaseEncoder(bundle.knowledge).fit(train_records, records)
        matrices = encoder.transform(records)
    available = np.stack(
        [matrices[name].getnnz(axis=1) > 0 for name in MODALITIES],
        axis=1,
    )
    relations = build_relation_pairs(bundle, ids)
    split_by_index = np.array([bundle.splits[disease] for disease in ids])
    with timed(f"Training one {config['embedding_dim']}-dimensional embedding model"):
        state, history, diagnostics = train_network(
            matrices,
            available,
            relations,
            split_by_index,
            config,
        )
    with timed("Encoding the full catalogue"):
        embeddings = encode_full(matrices, available, state, config)

    model = EmbeddingModel(
        ids=ids,
        names=[bundle.names[disease] for disease in ids],
        encoder=encoder,
        matrices=matrices,
        available=available,
        state=state,
        config=config,
        embeddings_full=embeddings,
        history=history,
        diagnostics=diagnostics,
    )
    with timed("Saving the production artifact"):
        model.save()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            embeddings,
            index=pd.Index(ids, name="orpha_id"),
            columns=[f"z_{index:03d}" for index in range(embeddings.shape[1])],
        )
        frame.to_parquet(MODEL_DIR / "embeddings.parquet")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EVALUATION_DIR / "training_history.json", history)
    summary = {
        "architecture": "sparse modality projection -> attention pooling -> normalized embedding -> cosine",
        "catalogue_size": len(ids),
        "embedding_dimension": embeddings.shape[1],
        "modalities": list(MODALITIES),
        "modality_coverage": {
            name: int(available[:, index].sum())
            for index, name in enumerate(MODALITIES)
        },
        "split_counts": {
            split: int((split_by_index == split).sum())
            for split in ("train", "validation", "test")
        },
        "relations": {
            "sibling": len(relations.sibling),
            "gene": len(relations.gene),
        },
        "config": config,
        **diagnostics,
    }
    write_json(EVALUATION_DIR / "training_summary.json", summary)
    print(f"[v4] saved {MODEL_DIR / 'embedding_model.joblib'}")
    print(f"[v4] best validation AUC {diagnostics['best_validation_mean_auc']:.4f}")
    print(
        "[v4] untouched test AUC "
        + ", ".join(f"{task}={value:.4f}" for task, value in diagnostics["test_auc"].items())
    )


if __name__ == "__main__":
    main()
