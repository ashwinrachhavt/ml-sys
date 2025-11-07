"""Lightweight file-based model registry for inference."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast


@dataclass
class RegistryEntry:
    """Metadata describing a registered model artifact."""

    identifier: str
    model_name: str
    path: Path
    metrics: dict[str, float]
    config: dict[str, Any]
    primary_metric: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "model_name": self.model_name,
            "path": str(self.path),
            "metrics": self.metrics,
            "config": self.config,
            "primary_metric": self.primary_metric,
            "created_at": self.created_at,
        }


class ModelRegistry:
    """Persist model artifacts and track the best-performing version."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                self._metadata = json.load(handle)
        else:
            self._metadata = {
                "models": [],
                "best_model_id": None,
                "primary_metric": None,
                "best_metric_value": None,
            }

    def _save_metadata(self) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self._metadata, handle, indent=2)

    def register_model(
        self,
        source_path: Path,
        *,
        model_name: str,
        metrics: dict[str, float],
        config: dict[str, Any],
        primary_metric: str,
    ) -> tuple[Path, bool]:
        """Store a copy of the model artifact and update best indicator.

        Returns the stored path and whether the model became the new champion.
        """

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        identifier = f"{model_name}-{timestamp}"
        dest_path = self.root / f"{identifier}.joblib"
        shutil.copy2(source_path, dest_path)

        entry = RegistryEntry(
            identifier=identifier,
            model_name=model_name,
            path=dest_path,
            metrics={k: float(v) for k, v in metrics.items()},
            config=config,
            primary_metric=primary_metric,
            created_at=datetime.utcnow().isoformat(),
        )
        self._metadata["models"].append(entry.to_dict())

        metric_value = entry.metrics.get(primary_metric)
        is_best = False
        if metric_value is not None:
            best_value = self._metadata.get("best_metric_value")
            if best_value is None or metric_value > best_value:
                self._metadata["best_model_id"] = identifier
                self._metadata["primary_metric"] = primary_metric
                self._metadata["best_metric_value"] = metric_value
                self._update_symlink(dest_path)
                is_best = True

        self._save_metadata()
        return dest_path, is_best

    def _update_symlink(self, target: Path) -> None:
        symlink = self.root / "best_model.joblib"
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        try:
            symlink.symlink_to(target.name)
        except OSError:  # pragma: no cover - fallback for filesystems without symlinks
            shutil.copy2(target, symlink)

    def get_best_model_path(self) -> Path:
        identifier = self._metadata.get("best_model_id")
        if not identifier:
            raise FileNotFoundError("No model has been registered yet.")
        path = self.root / f"{identifier}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Registered best model missing at {path}")
        return path

    def get_best_metadata(self) -> dict[str, Any]:
        identifier = self._metadata.get("best_model_id")
        if not identifier:
            raise FileNotFoundError("No model has been registered yet.")
        for entry in reversed(self._metadata["models"]):
            if entry["identifier"] == identifier:
                return cast(dict[str, Any], entry)
        raise FileNotFoundError("Best model metadata not found.")

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._metadata["models"])


__all__ = ["ModelRegistry"]
