"""Push/pull versioned pipeline outputs on the same HF dataset repo as raw.

Raw parquet already lives under ``data/`` in ``dataset.hf_repo_id``.
Upload ``interim/`` (and later ``processed/``) only — never re-push raw.

Push is opt-in: call ``push_dataset_tree`` from the CLI, or from the Airflow DAG
when ``PUSH_INTERIM`` / ``PUSH_PROCESSED`` is set. Preprocess and Spark never
upload unless that flag is enabled.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download, upload_folder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def load_full_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_dataset_repo_id(config_path: Path = CONFIG_PATH) -> tuple[str, str]:
    """Return (repo_id, repo_type) from dataset config (single project dataset repo)."""
    cfg = load_full_config(config_path)
    dataset = cfg["dataset"]
    return dataset["hf_repo_id"], dataset.get("hf_repo_type", "dataset")


def load_versioning_config(config_path: Path = CONFIG_PATH) -> dict:
    return load_full_config(config_path)["versioning"]


def push_dataset_tree(
    local_dir: Path | str,
    path_in_repo: str | None = None,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    config_path: Path = CONFIG_PATH,
    private: bool = False,
) -> str:
    """Upload ``local_dir`` under ``path_in_repo`` on the project dataset repo."""
    vcfg = load_versioning_config(config_path)
    default_repo_id, default_repo_type = resolve_dataset_repo_id(config_path)
    repo_id = repo_id or default_repo_id
    repo_type = repo_type or default_repo_type
    path_in_repo = path_in_repo or vcfg.get("path_in_repo", "interim")
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    api = HfApi()
    if not api.repo_exists(repo_id=repo_id, repo_type=repo_type):
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    result = upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        commit_message=f"Update {path_in_repo} from local {local_dir.name}",
    )
    logger.info("Pushed %s -> %s:%s (%s)", local_dir, repo_id, path_in_repo, result)
    return str(result)


def pull_dataset_tree(
    local_dir: Path | str,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    config_path: Path = CONFIG_PATH,
) -> Path:
    """Download a subset of the project dataset repo into ``local_dir``."""
    default_repo_id, default_repo_type = resolve_dataset_repo_id(config_path)
    repo_id = repo_id or default_repo_id
    repo_type = repo_type or default_repo_type
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    logger.info("Pulled %s@%s into %s", repo_id, revision or "main", local_dir)
    return local_dir


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Manual HF dataset versioning on the project dataset repo "
        "(push requires your approval to run)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    push_p = sub.add_parser("push", help="Upload a local tree (explicit only)")
    push_p.add_argument("local_dir", type=Path)
    push_p.add_argument(
        "path_in_repo",
        type=str,
        nargs="?",
        default=None,
        help="prefix in the HF dataset repo (default: config versioning.path_in_repo)",
    )
    push_p.add_argument("--repo-id", default=None)
    push_p.add_argument("--private", action="store_true")

    pull_p = sub.add_parser("pull", help="Download from the project dataset repo")
    pull_p.add_argument("local_dir", type=Path)
    pull_p.add_argument("--repo-id", default=None)
    pull_p.add_argument("--revision", default=None)
    pull_p.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
    )

    args = parser.parse_args()
    if args.command == "push":
        push_dataset_tree(
            args.local_dir,
            args.path_in_repo,
            repo_id=args.repo_id,
            private=args.private,
        )
    else:
        pull_dataset_tree(
            args.local_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            allow_patterns=args.allow_patterns,
        )


if __name__ == "__main__":
    main()
