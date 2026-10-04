"""Export the acquired SQLite evidence as a v1 weighted-Jaccard benchmark.

The output matches the aggregate CSV contract consumed by
experiments/01_compare_evaluation_literature_review.py.  It does not overwrite
the project's existing literature CSVs.  Automated ordinal scores are
normalized to [0, 1] and averaged within each paper/pair using extraction
confidence as a weight; NA dimensions remain absent rather than becoming zero.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / ".data/literature_acquisition/papers.sqlite"
DEFAULT_OUTPUT = (
    ROOT / ".data/experiments/v1_acquired/literature_disease_pairs.csv"
)
DIMENSION_LABELS = {
    "clinical_phenotype": "phenotype",
    "genetic_etiology": "genetic",
    "molecular_mechanism": "mechanism",
    "natural_history": "natural_history",
    "therapeutic_similarity": "therapeutic",
}
FIELDS = (
    "row_id",
    "disease_a",
    "disease_b",
    "disease_a_orpha_id",
    "disease_b_orpha_id",
    "similarity_score",
    "similarity_dimension",
    "relationship",
    "finding",
    "paper_title",
    "year",
    "link",
    "pmid",
    "evidence_checked",
    "extraction_confidence",
    "verification_status",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_link(row: sqlite3.Row) -> str:
    if row["pmcid"]:
        return f"https://europepmc.org/articles/{row['pmcid']}"
    if row["doi"]:
        return f"https://doi.org/{row['doi']}"
    if row["pmid"]:
        return f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/"
    urls = json.loads(row["source_urls_json"] or "[]")
    return str(urls[0]) if urls else ""


def export(database: Path, output: Path) -> dict[str, object]:
    if not database.is_file():
        raise FileNotFoundError(f"Acquisition database not found: {database}")
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    try:
        papers = {
            row["paper_pair_id"]: row
            for row in db.execute(
                """SELECT pp.paper_pair_id,pp.orpha_id_a,pp.orpha_id_b,
                p.pmid,p.pmcid,p.doi,p.title,p.year,p.source_urls_json,
                a.preferred_name AS disease_a,b.preferred_name AS disease_b
                FROM paper_pairs pp
                JOIN papers p ON p.paper_id=pp.paper_id
                JOIN orpha_entities a ON a.orpha_id=pp.orpha_id_a
                JOIN orpha_entities b ON b.orpha_id=pp.orpha_id_b
                ORDER BY pp.paper_pair_id"""
            )
        }
        scores: dict[int, list[sqlite3.Row]] = {}
        for row in db.execute(
            """SELECT paper_pair_id,dimension,ordinal_score,
            extraction_confidence,evidence_access_status,
            supporting_passage_or_paraphrase,verification_status
            FROM dimension_scores WHERE ordinal_score IS NOT NULL
            ORDER BY paper_pair_id,dimension"""
        ):
            scores.setdefault(row["paper_pair_id"], []).append(row)

        rows = []
        for paper_pair_id, paper in papers.items():
            observations = scores.get(paper_pair_id, [])
            if not observations:
                continue
            total_weight = sum(row["extraction_confidence"] for row in observations)
            normalized = sum(
                (row["ordinal_score"] / 4.0) * row["extraction_confidence"]
                for row in observations
            ) / total_weight
            dimensions = sorted(
                DIMENSION_LABELS[row["dimension"]] for row in observations
            )
            access = (
                "full text"
                if any(
                    row["evidence_access_status"] == "full_text"
                    for row in observations
                )
                else "abstract only"
            )
            max_confidence = max(
                row["extraction_confidence"] for row in observations
            )
            evidence = " | ".join(
                str(row["supporting_passage_or_paraphrase"])
                for row in observations[:2]
            )
            rows.append(
                {
                    "row_id": len(rows) + 1,
                    "disease_a": paper["disease_a"],
                    "disease_b": paper["disease_b"],
                    "disease_a_orpha_id": paper["orpha_id_a"],
                    "disease_b_orpha_id": paper["orpha_id_b"],
                    "similarity_score": f"{normalized:.8f}",
                    "similarity_dimension": "; ".join(dimensions),
                    "relationship": "automated unverified",
                    "finding": evidence,
                    "paper_title": paper["title"],
                    "year": paper["year"] or "",
                    "link": source_link(paper),
                    "pmid": paper["pmid"] or "",
                    "evidence_checked": access,
                    "extraction_confidence": max_confidence,
                    "verification_status": "unverified",
                }
            )
    finally:
        db.close()

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "generated_at": utc_now(),
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "rows": len(rows),
        "unique_pairs": len(
            {
                tuple(
                    sorted(
                        (row["disease_a_orpha_id"], row["disease_b_orpha_id"])
                    )
                )
                for row in rows
            }
        ),
        "score_transform": (
            "Within each paper/pair, non-NA ordinal scores are divided by 4 "
            "and averaged with extraction_confidence as the weight."
        ),
        "missing_policy": "NA dimensions are omitted, never converted to zero.",
        "verification_status": "All source scores are automated and unverified.",
        "dimensions": list(DIMENSION_LABELS),
    }
    manifest_path = output.with_name("export_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export(args.database.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
