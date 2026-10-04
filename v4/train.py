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
from fusion import CosineEngine, fit_fusion, pair_score_scales
from modalities import MODALITIES, TASK_MASKS, DiseaseEncoder
from model import (
    DEFAULT_CONFIG,
    EmbeddingModel,
    build_relation_pairs,
    choose_fusion_mix,
    encode_full,
    encode_masked,
    split_pairs,
    train_network,
    union_known_neighbours,
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
    relations = build_relation_pairs(
        bundle,
        ids,
        phenotype_matrix=matrices["phenotype"],
        phenotype_neighbours=int(config["phenotype_neighbours"]),
        phenotype_min_similarity=float(config["phenotype_min_similarity"]),
    )
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

    engine = CosineEngine(matrices, available)
    train_pairs = {
        task: split_pairs(pairs, split_by_index)["train"]
        for task, pairs in relations.by_task().items()
    }
    with timed("Fitting v2-style cosine fusion on train/train pairs"):
        fusion = fit_fusion(
            engine,
            train_pairs,
            np.flatnonzero(split_by_index == "train"),
            union_known_neighbours(relations, len(ids)),
            config,
        )
        rng = np.random.default_rng(int(config["seed"]))
        train_index = np.flatnonzero(split_by_index == "train")
        a = train_index[rng.integers(0, len(train_index), 20_000)]
        b = train_index[rng.integers(0, len(train_index), 20_000)]
        keep = a != b
        embedding_scale, fusion_scale = pair_score_scales(embeddings, engine, fusion, a[keep], b[keep])
        embeddings_by_task = {
            task: encode_masked(matrices, available, state, config, TASK_MASKS[task])
            for task in ("sibling", "gene")
        }
        fusion_mix, mix_scores = choose_fusion_mix(
            engine,
            fusion,
            embeddings_by_task,
            relations,
            split_by_index,
            embedding_scale,
            fusion_scale,
            config["fusion_mixes"],
        )
        diagnostics["fusion"] = {
            "coefficients": [
                {
                    "modality": name,
                    "similarity_weight": float(weight),
                    "availability_weight": float(offset),
                }
                for name, weight, offset in zip(
                    fusion.modalities, fusion.similarity_weights, fusion.availability_weights
                )
            ],
            "embedding_scale": embedding_scale,
            "fusion_scale": fusion_scale,
            "fusion_mix": fusion_mix,
            "validation_map_by_mix": mix_scores,
        }

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
        fusion=fusion,
        embedding_scale=embedding_scale,
        fusion_scale=fusion_scale,
        fusion_mix=fusion_mix,
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
        "architecture": "v4 embedding cosine + v2-style non-negative modality fusion",
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
            task: len(pairs)
            for task, pairs in relations.by_task().items()
        },
        "config": config,
        **diagnostics,
    }
    write_json(EVALUATION_DIR / "training_summary.json", summary)
    print(f"[v4] saved {MODEL_DIR / 'embedding_model.joblib'}")
    print(f"[v4] best validation MAP {diagnostics['best_validation_mean_map']:.4f}")
    print(f"[v4] fusion mix {fusion_mix:.2f} (0=embedding, 1=v2-style fusion)")
    print(
        "[v4] untouched test MAP "
        + ", ".join(
            f"{task}={value['map']:.4f}"
            for task, value in diagnostics["test_retrieval"].items()
        )
    )


if __name__ == "__main__":
    main()
