"""Pipeline configuration dataclass."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the ABMIL slide embedding pipeline.

    Groups all tunable parameters: tile extraction, quality gate thresholds,
    feature extraction, ABMIL aggregation, visualization, and output settings.
    """

    # Tile extraction
    tile_size: int = 224
    magnification_level: int = 0  # OpenSlide level
    max_tiles: Optional[int] = None  # None = no limit

    # Quality gate thresholds
    max_mean_intensity: float = 240.0  # above → background
    min_tissue_pct: float = 0.50  # below → insufficient tissue
    tissue_intensity_threshold: int = 220  # grayscale cutoff for tissue pixel
    min_stain_saturation: float = 20.0  # HSV saturation floor (0-255)
    min_pixel_std: float = 10.0  # below → uninformative

    # Feature extraction
    model_name: str = "bioptimus/H-optimus-0"
    batch_size: int = 32  # tiles per inference batch
    device: str = "auto"  # "auto", "cpu", "cuda"

    # ABMIL
    attention_hidden_dim: int = 128  # L in the paper
    projection_dim: Optional[int] = None  # None = no projection

    # Visualization
    top_k: int = 10

    # Output
    output_csv: str = "slide_embeddings.csv"
