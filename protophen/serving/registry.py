"""
Model versioning and registry for ProToPhen.

Provides a lightweight, filesystem-backed model registry that tracks
checkpoint versions, associated metadata, and supports rollback.

Session 10.2 additions:
- ``register_from_trainer_checkpoint()``: reads Trainer/CheckpointCallback
  checkpoints and extracts metrics, epoch, and config automatically.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RegistryConfig:
    """Configuration for the model registry."""

    registry_dir: str = "./model_registry"
    max_versions: int = 20
    metadata_filename: str = "registry.json"


# =============================================================================
# Model Version
# =============================================================================

@dataclass
class ModelVersion:
    """Metadata for a single registered model version."""

    version: str
    checkpoint_path: str
    registered_at: str  # ISO-8601
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    stage: Literal["staging", "production", "archived"] = "staging"
    parent_version: Optional[str] = None
    epoch: Optional[int] = None
    trainer_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "registered_at": self.registered_at,
            "description": self.description,
            "metrics": self.metrics,
            "config": self.config,
            "tags": self.tags,
            "stage": self.stage,
            "parent_version": self.parent_version,
            "epoch": self.epoch,
            "trainer_config": self.trainer_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelVersion:
        # Handle older registry entries that lack new fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """
    Filesystem-backed model version registry.

    Versions are stored under ``registry_dir/`` with the layout::

        registry_dir/
        ├── registry.json
        ├── v1/
        │   └── model.pt
        ├── v2/
        │   └── model.pt
        └── ...

    Example::

        registry = ModelRegistry("./model_registry")

        # Register from a Trainer checkpoint
        registry.register_from_trainer_checkpoint("checkpoints/best_model.pt")

        # Or register manually
        registry.register(
            checkpoint_path="checkpoints/best.pt",
            version="v1",
            metrics={"val_r2": 0.72},
        )

        # Promote to production
        registry.set_stage("v1", "production")

        # Load the production model
        path = registry.get_production_checkpoint()
    """

    def __init__(
        self,
        config: Optional[RegistryConfig] = None,
        registry_dir: Optional[str] = None,
    ):
        if config is None:
            config = RegistryConfig()
        if registry_dir is not None:
            config.registry_dir = registry_dir

        self.config = config
        self._dir = Path(config.registry_dir)
        self._index_path = self._dir / config.metadata_filename
        self._versions: Dict[str, ModelVersion] = {}

        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

        logger.info(
            f"ModelRegistry at {self._dir} ({len(self._versions)} versions)"
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_index(self) -> None:
        if self._index_path.exists():
            with open(self._index_path) as f:
                data = json.load(f)
            self._versions = {
                k: ModelVersion.from_dict(v)
                for k, v in data.get("versions", {}).items()
            }
        else:
            self._versions = {}

    def _save_index(self) -> None:
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "versions": {k: v.to_dict() for k, v in self._versions.items()},
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        checkpoint_path: str | Path,
        version: Optional[str] = None,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        stage: Literal["staging", "production", "archived"] = "staging",
        parent_version: Optional[str] = None,
        copy_checkpoint: bool = True,
        epoch: Optional[int] = None,
        trainer_config: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            checkpoint_path: Path to the checkpoint file.
            version: Version label (auto-generated if ``None``).
            description: Human-readable description.
            metrics: Evaluation metrics to store.
            config: Model / training config snapshot.
            tags: Searchable tags.
            stage: Initial lifecycle stage.
            parent_version: Version this was derived from.
            copy_checkpoint: Copy checkpoint into registry directory.
            epoch: Training epoch (for provenance).
            trainer_config: TrainerConfig dict (for reproducibility).

        Returns:
            The newly created ``ModelVersion``.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if version is None:
            existing = [
                v.version for v in self._versions.values()
                if v.version.startswith("v")
            ]
            nums = []
            for v in existing:
                try:
                    nums.append(int(v[1:]))
                except ValueError:
                    pass
            next_num = max(nums, default=0) + 1
            version = f"v{next_num}"

        if version in self._versions:
            raise ValueError(
                f"Version '{version}' already exists. Use a different label or "
                f"delete the existing version first."
            )

        version_dir = self._dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        if copy_checkpoint:
            dest = version_dir / "model.pt"
            shutil.copy2(checkpoint_path, dest)
            stored_path = str(dest)
        else:
            stored_path = str(checkpoint_path.resolve())

        mv = ModelVersion(
            version=version,
            checkpoint_path=stored_path,
            registered_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            metrics=metrics or {},
            config=config or {},
            tags=tags or [],
            stage=stage,
            parent_version=parent_version,
            epoch=epoch,
            trainer_config=trainer_config,
        )

        self._versions[version] = mv
        self._enforce_max_versions()
        self._save_index()

        logger.info(f"Registered model version '{version}' (stage={stage})")
        return mv

    def register_from_trainer_checkpoint(
        self,
        checkpoint_path: str | Path,
        version: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        stage: Literal["staging", "production", "archived"] = "staging",
        copy_checkpoint: bool = True,
    ) -> ModelVersion:
        """
        Register a checkpoint produced by ``Trainer.save_checkpoint()``
        or ``CheckpointCallback`` (Session 6).

        This convenience method loads the checkpoint, extracts the epoch,
        metrics (``best_val_loss``, ``best_value``), and ``TrainerConfig``,
        then delegates to :meth:`register`.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint.
            version: Version label (auto-generated if ``None``).
            description: Human-readable description.
            tags: Searchable tags.
            stage: Initial lifecycle stage.
            copy_checkpoint: Copy checkpoint into registry directory.

        Returns:
            The newly created ``ModelVersion``.
        """
        from protophen.serving.pipeline import load_checkpoint

        checkpoint = load_checkpoint(str(checkpoint_path), device="cpu")

        # Extract metrics
        metrics: Dict[str, float] = {}
        if "best_val_loss" in checkpoint:
            metrics["best_val_loss"] = float(checkpoint["best_val_loss"])
        if "best_value" in checkpoint:
            monitor = checkpoint.get("monitor", "val_loss")
            metrics[f"best_{monitor}"] = float(checkpoint["best_value"])
        # Merge any normalised metrics
        metrics.update(checkpoint.get("metrics", {}))

        # Extract epoch
        epoch = checkpoint.get("epoch")

        # Extract configs
        model_config = checkpoint.get("config", {})
        if isinstance(model_config, dict):
            model_config_dict = model_config
        else:
            model_config_dict = {}

        trainer_config = checkpoint.get("_trainer_config")

        # Auto-generate description if empty
        if not description and epoch is not None:
            best_metric = metrics.get("best_val_loss", metrics.get("best_val_loss"))
            desc_parts = [f"Trainer checkpoint at epoch {epoch}"]
            if best_metric is not None:
                desc_parts.append(f"best_val_loss={best_metric:.4f}")
            description = ", ".join(desc_parts)

        return self.register(
            checkpoint_path=checkpoint_path,
            version=version,
            description=description,
            metrics=metrics,
            config=model_config_dict,
            tags=tags or ["trainer_checkpoint"],
            stage=stage,
            copy_checkpoint=copy_checkpoint,
            epoch=epoch,
            trainer_config=trainer_config if isinstance(trainer_config, dict) else None,
        )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_version(self, version: str) -> ModelVersion:
        if version not in self._versions:
            raise KeyError(f"Version '{version}' not found in registry.")
        return self._versions[version]

    def list_versions(
        self,
        stage: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        versions = list(self._versions.values())
        if stage is not None:
            versions = [v for v in versions if v.stage == stage]
        if tags:
            tag_set = set(tags)
            versions = [v for v in versions if tag_set.issubset(set(v.tags))]
        versions.sort(key=lambda v: v.registered_at, reverse=True)
        return versions

    def get_latest(self, stage: Optional[str] = None) -> Optional[ModelVersion]:
        versions = self.list_versions(stage=stage)
        return versions[0] if versions else None

    def get_production_checkpoint(self) -> Optional[str]:
        prod = self.get_latest(stage="production")
        return prod.checkpoint_path if prod else None

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def set_stage(
        self,
        version: str,
        stage: Literal["staging", "production", "archived"],
    ) -> None:
        mv = self.get_version(version)
        if stage == "production":
            for v in self._versions.values():
                if v.stage == "production" and v.version != version:
                    v.stage = "archived"
                    logger.info(
                        f"Version '{v.version}' archived (replaced by '{version}')"
                    )
        mv.stage = stage
        self._save_index()
        logger.info(f"Version '{version}' → stage={stage}")

    def delete_version(self, version: str) -> None:
        self.get_version(version)  # validate existence
        version_dir = self._dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        del self._versions[version]
        self._save_index()
        logger.info(f"Deleted version '{version}'")

    def _enforce_max_versions(self) -> None:
        while len(self._versions) > self.config.max_versions:
            archived = sorted(
                [v for v in self._versions.values() if v.stage == "archived"],
                key=lambda v: v.registered_at,
            )
            if not archived:
                logger.warning(
                    f"Registry has {len(self._versions)} versions "
                    f"(limit {self.config.max_versions}) but no archived "
                    f"versions to evict."
                )
                break
            oldest = archived[0]
            logger.info(
                f"Evicting oldest archived version '{oldest.version}' "
                f"(registered {oldest.registered_at})"
            )
            self.delete_version(oldest.version)

    # =========================================================================
    # Comparison / A-B Testing Support
    # =========================================================================

    def compare_versions(
        self, version_a: str, version_b: str,
    ) -> Dict[str, Any]:
        mv_a = self.get_version(version_a)
        mv_b = self.get_version(version_b)
        all_metrics = set(mv_a.metrics.keys()) | set(mv_b.metrics.keys())
        comparison: Dict[str, Any] = {}
        for metric in sorted(all_metrics):
            val_a = mv_a.metrics.get(metric)
            val_b = mv_b.metrics.get(metric)
            entry: Dict[str, Any] = {version_a: val_a, version_b: val_b}
            if val_a is not None and val_b is not None:
                entry["delta"] = val_b - val_a
                entry["relative_change"] = (
                    (val_b - val_a) / abs(val_a) if val_a != 0 else None
                )
            comparison[metric] = entry
        return {"version_a": version_a, "version_b": version_b, "metrics": comparison}

    def get_best_version(
        self, metric: str, higher_is_better: bool = True,
        stage: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        candidates = self.list_versions(stage=stage)
        candidates = [v for v in candidates if metric in v.metrics]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda v: v.metrics[metric],
            reverse=higher_is_better,
        )[0]

    # =========================================================================
    # Rollback
    # =========================================================================

    def rollback(self) -> Optional[ModelVersion]:
        archived = sorted(
            [v for v in self._versions.values() if v.stage == "archived"],
            key=lambda v: v.registered_at,
            reverse=True,
        )
        if not archived:
            logger.warning("No archived versions available for rollback.")
            return None
        rollback_target = archived[0]
        self.set_stage(rollback_target.version, "production")
        logger.info(f"Rolled back to version '{rollback_target.version}'")
        return rollback_target

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        stage_counts: Dict[str, int] = {}
        for v in self._versions.values():
            stage_counts[v.stage] = stage_counts.get(v.stage, 0) + 1
        production = self.get_latest(stage="production")
        return {
            "registry_dir": str(self._dir),
            "total_versions": len(self._versions),
            "stages": stage_counts,
            "production_version": production.version if production else None,
            "latest_version": (
                self.get_latest().version if self._versions else None
            ),
        }

    def __repr__(self) -> str:
        prod = self.get_latest(stage="production")
        prod_str = prod.version if prod else "none"
        return (
            f"ModelRegistry(dir={self._dir}, "
            f"versions={len(self._versions)}, "
            f"production={prod_str})"
        )