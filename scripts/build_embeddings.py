"""Script to parse Q/A pairs from Excel and upsert embeddings into Qdrant."""
from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.config import get_settings
from backend.embedder import Embedder, EmbeddingError


@dataclass
class QAPair:
    question: str
    answer: str
    group_id: int
    source: str


def load_qa_pairs(xlsx_path: Path) -> List[QAPair]:
    """Read the Excel file and extract question/answer pairs."""

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")

    col_b = df.iloc[:, 1] if df.shape[1] > 1 else pd.Series(dtype=float)
    col_c = df.iloc[:, 2] if df.shape[1] > 2 else pd.Series(dtype=str)

    qa_pairs: List[QAPair] = []
    row_indices = col_c.index.tolist()

    idx = 0
    while idx < len(row_indices):
        row_idx = row_indices[idx]
        cell_value = col_c.iloc[idx]

        if isinstance(cell_value, str):
            value = cell_value.strip()
        else:
            value = ""

        if value.upper().startswith("Q."):
            question = value[2:].strip()
            if not question:
                raise ValueError(f"Empty question detected at row C{row_idx + 1}")

            if idx + 1 >= len(row_indices):
                raise ValueError(f"Missing answer for question at row C{row_idx + 1}")

            next_row_idx = row_indices[idx + 1]
            answer_cell = col_c.iloc[idx + 1]
            answer_value = answer_cell.strip() if isinstance(answer_cell, str) else ""

            if not answer_value.upper().startswith("A."):
                raise ValueError(
                    "Expected answer after question at row C"
                    f"{row_idx + 1}, found value '{answer_cell}' at row C{next_row_idx + 1}"
                )

            answer = answer_value[2:].strip()
            if not answer:
                raise ValueError(f"Empty answer detected at row C{next_row_idx + 1}")

            group_raw = col_b.loc[row_idx] if row_idx in col_b.index else None
            if pd.isna(group_raw):
                group_raw = col_b.loc[next_row_idx] if next_row_idx in col_b.index else None
            try:
                group_id = int(group_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid group id in column B for rows {row_idx + 1}/{next_row_idx + 1}: {group_raw}"
                ) from None

            source = f"xlsx_row_C{row_idx + 1}_C{next_row_idx + 1}"
            qa_pairs.append(QAPair(question=question, answer=answer, group_id=group_id, source=source))

            idx += 2
        else:
            idx += 1

    if len(qa_pairs) == 0:
        raise ValueError("No question/answer pairs were extracted from the workbook")

    return qa_pairs


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Ensure the target collection exists with the required configuration."""

    try:
        client.get_collection(collection_name)
    except UnexpectedResponse:
        logging.info("Creating collection '%s'", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE),
        )


def build_points(pairs: List[QAPair], embedder: Embedder) -> List[qmodels.PointStruct]:
    """Create Qdrant point structures for the provided Q/A pairs."""

    points: List[qmodels.PointStruct] = []
    for pair in pairs:
        embedding = embedder.embed(pair.question)
        point_id = hashlib.md5(pair.question.encode("utf-8")).hexdigest()
        payload = {
            "question": pair.question,
            "answer": pair.answer,
            "source": pair.source,
            "group_id": pair.group_id,
        }
        points.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
    return points


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    xlsx_path = Path("data/QA_data.xlsx")

    try:
        qa_pairs = load_qa_pairs(xlsx_path)
    except Exception as exc:
        logging.error("Failed to load Q/A pairs: %s", exc)
        return 1

    logging.info("Extracted %s Q/A pairs", len(qa_pairs))

    try:
        settings = get_settings()
        embedder = Embedder()
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        ensure_collection(client, settings.qdrant_collection)
        points = build_points(qa_pairs, embedder)
        if points:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
    except EmbeddingError as exc:
        logging.error("Embedding generation failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - network operations
        logging.error("Failed to upsert embeddings: %s", exc)
        return 1

    logging.info("Upserted %s points into collection '%s'", len(qa_pairs), settings.qdrant_collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
