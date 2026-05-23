from __future__ import annotations

import gzip
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    domain: str
    expected_role: str


def candidate_datasets(data_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            "criteo_uplift",
            data_dir / "criteo" / "criteo-research-uplift-v2.1.csv.gz",
            "ads incrementality",
            "Ads treatment-control benchmark with visit, conversion, exposure, and treatment.",
        ),
        DatasetSpec(
            "open_bandit",
            data_dir / "open_bandit" / "open_bandit_dataset.zip",
            "recommendation bandit",
            "Known-propensity logged recommendation data.",
        ),
        DatasetSpec(
            "kuairand",
            data_dir / "KuaiRand" / "10439422.zip",
            "member-experience recommendation",
            "Randomly exposed sequential short-video recommendation data.",
        ),
        DatasetSpec(
            "movielens_32m",
            data_dir / "movieLens" / "ml-32m.zip",
            "recommendation graph",
            "Large user-item rating graph for semi-synthetic interference simulations.",
        ),
        DatasetSpec(
            "mind_large",
            data_dir / "MIND" / "MINDlarge_train.zip",
            "news recommendation",
            "News recommendation data for optional member-experience robustness.",
        ),
        DatasetSpec(
            "mind_small",
            data_dir / "MIND-small" / "MINDsmall_train.zip",
            "news recommendation",
            "Small MIND version for fast local smoke tests.",
        ),
        DatasetSpec(
            "kuairec",
            data_dir / "Kuairec" / "18164998.zip",
            "recommendation",
            "KuaiRec data for optional recommendation robustness.",
        ),
    ]


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024**2), 2) if path.exists() else 0.0


def _member_preview(path: Path, limit: int = 6) -> str:
    if not path.exists():
        return ""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return "; ".join(zf.namelist()[:limit])
    if path.suffix == ".gz" and not path.name.endswith(".tar.gz"):
        with gzip.open(path, "rt") as handle:
            return handle.readline().strip()
    return ""


def dataset_readiness_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in candidate_datasets(data_dir):
        rows.append(
            {
                "dataset": spec.name,
                "domain": spec.domain,
                "present": spec.path.exists(),
                "size_mb": _size_mb(spec.path),
                "path": str(spec.path),
                "expected_role": spec.expected_role,
                "member_or_header_preview": _member_preview(spec.path),
            }
        )
    return pd.DataFrame(rows)


def kuairand_archive_members(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame(columns=["member", "compressed_mb", "raw_mb"])
    with zipfile.ZipFile(zip_path) as zf:
        rows = [
            {
                "member": info.filename,
                "compressed_mb": round(info.compress_size / (1024**2), 2),
                "raw_mb": round(info.file_size / (1024**2), 2),
            }
            for info in zf.infolist()
        ]
    return pd.DataFrame(rows)


def open_bandit_archive_members(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame(columns=["member", "compressed_mb", "raw_mb"])
    with zipfile.ZipFile(zip_path) as zf:
        rows = [
            {
                "member": info.filename,
                "compressed_mb": round(info.compress_size / (1024**2), 2),
                "raw_mb": round(info.file_size / (1024**2), 2),
            }
            for info in zf.infolist()
        ]
    return pd.DataFrame(rows)


def kuairand_nested_members(
    zip_path: Path,
    nested_name: str = "KuaiRand-Pure.tar.gz",
) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame(columns=["member", "size_mb"])
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(nested_name) as nested:
            with tarfile.open(fileobj=nested, mode="r:gz") as tf:
                rows = [
                    {"member": m.name, "size_mb": round(m.size / (1024**2), 3)}
                    for m in tf.getmembers()
                    if m.isfile()
                ]
    return pd.DataFrame(rows)
