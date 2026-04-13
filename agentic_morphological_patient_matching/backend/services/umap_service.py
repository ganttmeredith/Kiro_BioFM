"""UMAPService: runs UMAP projections, assigns clusters, and caches results."""

import threading
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from umap_retrieval.data import apply_metadata_filters
from umap_retrieval.embedding import assign_clusters_by_metadata, run_umap
from umap_retrieval.visualization import plot_umap_interactive

from backend.models import UMAPParams, UMAPResponse

# Cache key: (n_neighbors, min_dist, color_by, frozenset of filter items)
_CacheKey = Tuple[int, float, str, FrozenSet]


def _make_cache_key(params: UMAPParams) -> _CacheKey:
    """Build a hashable cache key from UMAP params (excludes highlight params)."""
    filter_frozen = frozenset(
        (k, tuple(sorted(v))) for k, v in params.filters.items()
    )
    return (params.n_neighbors, params.min_dist, params.color_by, filter_frozen)


def _make_coords_key(params: UMAPParams) -> Tuple[int, float, FrozenSet]:
    """Cache key for just the UMAP coordinates (excludes color_by)."""
    filter_frozen = frozenset(
        (k, tuple(sorted(v))) for k, v in params.filters.items()
    )
    return (params.n_neighbors, params.min_dist, filter_frozen)


class UMAPService:
    """Runs UMAP projections and caches results keyed by (n_neighbors, min_dist, filters)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[_CacheKey, UMAPResponse] = {}
        # Store raw umap_df keyed by coords-only key so color_by changes can reuse it
        self._umap_df_cache: Dict[Tuple, object] = {}
        self._last_key: Optional[_CacheKey] = None

    def run_umap(self, params: UMAPParams, data_service) -> UMAPResponse:
        """Project embeddings to 2D, assign clusters, and return a Plotly figure.

        Results are cached by (n_neighbors, min_dist, frozenset(filters)).
        Identical subsequent calls return the cached result without recomputation.

        Args:
            params: UMAP run parameters (n_neighbors, min_dist, color_by, filters).
            data_service: DataService instance providing embeddings and metadata.

        Returns:
            UMAPResponse with plotly_json, n_points, and n_clusters.
        """
        key = _make_cache_key(params)
        coords_key = _make_coords_key(params)

        with self._lock:
            # If highlight params provided, re-render from cached umap_df without caching result
            has_highlights = bool(params.query_slide or params.neighbor_slides)

            # Full cache hit (same coords + same color_by, no highlights)
            if not has_highlights and key in self._cache:
                self._last_key = key
                return self._cache[key]

            embeddings: Dict[str, np.ndarray] = dict(data_service.get_embeddings())
            metadata_df = data_service.get_metadata()

            # Apply metadata filters to restrict the slide pool
            if params.filters:
                filtered_meta = apply_metadata_filters(metadata_df, params.filters)
                filtered_names = set(filtered_meta["slide_name"])
                embeddings = {k: v for k, v in embeddings.items() if k in filtered_names}

            # Reuse existing umap_df if only color_by or highlights changed
            if coords_key in self._umap_df_cache:
                umap_df = self._umap_df_cache[coords_key]
            else:
                umap_df = run_umap(embeddings, params.n_neighbors, params.min_dist)
                self._umap_df_cache[coords_key] = umap_df

            umap_df = assign_clusters_by_metadata(umap_df, metadata_df, params.color_by)

            fig = plot_umap_interactive(
                umap_df,
                metadata_df,
                color_by=params.color_by,
                query_slide=params.query_slide or None,
                neighbors=params.neighbor_slides or None,
            )
            plotly_json = fig.to_json()

            result = UMAPResponse(
                plotly_json=plotly_json,
                n_points=len(umap_df),
                n_clusters=int(umap_df["cluster"].nunique()),
            )

            # Only cache non-highlight results
            if not has_highlights:
                self._cache[key] = result
            self._last_key = key

        return result

    def get_cached_umap(self) -> Optional[UMAPResponse]:
        """Return the most recently computed UMAPResponse, or None if not yet run."""
        if self._last_key is None:
            return None
        return self._cache.get(self._last_key)


# Module-level singleton shared by all routers
umap_service = UMAPService()
