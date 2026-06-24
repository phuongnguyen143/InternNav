"""Load merged base + scene YAML configuration."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_ACTIVE: "VlnConfig | None" = None
_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_data_root(data: dict) -> str:
    value = os.environ.get("VLN_DATA_ROOT") or data.get("data_root", "")
    return str(Path(value).expanduser()) if value else ""


def _build_var_map(data: dict) -> dict[str, str]:
    var_map: dict[str, str] = {}
    data_root = _resolve_data_root(data)
    if data_root:
        var_map["data_root"] = data_root
    for key, value in data.items():
        if key == "paths" or isinstance(value, (dict, list)):
            continue
        if isinstance(value, str):
            var_map[key] = value
    return var_map


def _expand_string(value: str, var_map: dict[str, str]) -> str:
    prev = None
    out = value
    while out != prev:
        prev = out

        def repl(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            if name not in var_map:
                raise KeyError(f"Unknown config variable: {name}")
            return var_map[name]

        out = _VAR_PATTERN.sub(repl, out)
    return os.path.expanduser(os.path.expandvars(out))


def _expand_vars(node: Any, var_map: dict[str, str]) -> Any:
    if isinstance(node, dict):
        return {key: _expand_vars(value, var_map) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_vars(value, var_map) for value in node]
    if isinstance(node, str):
        return _expand_string(node, var_map)
    return node


def _finalize_config(data: dict) -> dict:
    data = deepcopy(data)
    data_root = _resolve_data_root(data)
    if data_root:
        data["data_root"] = data_root
    var_map = _build_var_map(data)
    return _expand_vars(data, var_map)


def _resolve_scene_path(scene: str | Path) -> Path:
    scene_path = Path(scene)
    if scene_path.is_file():
        return scene_path
    if scene_path.suffix in (".yaml", ".yml"):
        candidate = _CONFIGS_DIR / "scenes" / scene_path.name
        if candidate.is_file():
            return candidate
    candidate = _CONFIGS_DIR / "scenes" / f"{scene}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Scene config not found: {scene}")


class VlnConfig:
    """Merged VLN pipeline configuration."""

    def __init__(self, data: dict, scene_name: str = ""):
        self._data = data
        self.scene_name = scene_name or data.get("scene", "")

    @property
    def raw(self) -> dict:
        return self._data

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, key: str, default: str | None = None) -> Path:
        value = self.get("paths", key, default=default)
        if value is None:
            raise KeyError(f"paths.{key} not set in config (scene={self.scene_name})")
        return Path(value).expanduser()

    @property
    def filenames(self) -> dict:
        return self.get("filenames", default={})

    @property
    def robot(self) -> dict:
        return self.get("robot", default={})

    @property
    def ros(self) -> dict:
        return self.get("ros", default={})

    @property
    def keyframe(self) -> dict:
        return self.get("keyframe", default={})

    @property
    def depth(self) -> dict:
        return self.get("depth", default={})

    @property
    def camera(self) -> dict:
        return self.get("camera", default={})

    @property
    def slam_path(self) -> dict:
        return self.get("slam_path", default={})

    @property
    def precompute(self) -> dict:
        return self.get("precompute", default={})

    def data_root(self) -> Path:
        value = self.get("data_root", default="")
        if not value:
            raise KeyError("data_root not set in config (set in base.yaml or VLN_DATA_ROOT)")
        return Path(value)

    def floor_trajectory_filename(self) -> str:
        return str(self.filenames.get("floor_trajectory", "floor_trajectory.txt"))

    def floor_calibration_filename(self) -> str:
        return str(self.filenames.get("floor_calibration", "floor_calibration.json"))


def load_config(scene: str | Path | None = None) -> VlnConfig:
    """Load base.yaml merged with a scene file or explicit yaml path."""
    base_path = _CONFIGS_DIR / "base.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base config: {base_path}")

    with base_path.open() as f:
        data = yaml.safe_load(f) or {}

    scene_name = ""
    if scene is not None:
        scene_path = _resolve_scene_path(scene)
        with scene_path.open() as f:
            scene_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, scene_data)
        scene_name = str(scene_data.get("scene", scene_path.stem))
    elif os.environ.get("VLN_SCENE"):
        return load_config(os.environ["VLN_SCENE"])

    data = _finalize_config(data)
    cfg = VlnConfig(data, scene_name=scene_name)
    global _ACTIVE
    _ACTIVE = cfg
    return cfg


def get_config() -> VlnConfig:
    """Return the last loaded config or load base-only defaults."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config()
    return _ACTIVE
