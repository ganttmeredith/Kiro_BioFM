"""CSV I/O utilities and logging setup."""

import csv
import logging
import os
from typing import Dict

import numpy as np


logger = logging.getLogger("abmil_pipeline")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the abmil_pipeline logger with a console handler.

    Args:
        level: Logging level (default INFO).
    """
    _logger = logging.getLogger("abmil_pipeline")
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    _logger.setLevel(level)


def save_embedding_to_csv(
    csv_path: str,
    slide_name: str,
    embedding: np.ndarray,
) -> None:
    """Append one embedding row to a CSV file.

    Creates the file with a header if it doesn't exist. Each row contains:
    slide_name, embedding_dim, e_0, e_1, ..., e_{dim-1}

    Args:
        csv_path: Path to the output CSV file.
        slide_name: Identifier for the slide.
        embedding: 1-D numpy array of the embedding vector.
    """
    embedding = np.asarray(embedding).ravel()
    dim = len(embedding)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["slide_name", "embedding_dim"] + [
                f"e_{i}" for i in range(dim)
            ]
            writer.writerow(header)
        row = [slide_name, dim] + [float(v) for v in embedding]
        writer.writerow(row)


def load_embeddings_from_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """Read all embeddings from a CSV file.

    Args:
        csv_path: Path to the CSV file written by save_embedding_to_csv.

    Returns:
        Dictionary mapping slide_name to its embedding as a numpy array.
    """
    embeddings: Dict[str, np.ndarray] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            slide_name = row[0]
            dim = int(row[1])
            values = [float(v) for v in row[2 : 2 + dim]]
            embeddings[slide_name] = np.array(values, dtype=np.float64)
    return embeddings
