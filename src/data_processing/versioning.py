"""Push/pull versioned pipeline outputs on HF Hub.

Raw parquet already lives under ``data/`` in ``dataset.hf_repo_id``.
Upload ``interim/`` (and later ``processed/``) only — never re-push raw.

Trained artifacts go to a separate model-type repo
(``versioning.hf_model_repo_id``): first the four model dirs, then
``winner.json`` + ``winner/``.

Push is opt-in: call the CLI, or the Airflow DAG when
``versioning.push_interim`` / ``push_processed`` / ``push_models`` is true.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download, upload_file, upload_folder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
TRAINED_MODEL_DIRS = ("rf", "xgboost", "lightgbm", "resnet18")


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


def versioning_push_enabled(key: str, config_path: Path = CONFIG_PATH) -> bool:
    """Return whether ``versioning.<key>`` is enabled (e.g. ``push_interim``)."""
    return bool(load_versioning_config(config_path).get(key, False))


def config_enabled(
    section: str,
    key: str = "enabled",
    *,
    default: bool = True,
    config_path: Path = CONFIG_PATH,
) -> bool:
    """Return a boolean flag from any config section (missing key uses ``default``)."""
    cfg = load_full_config(config_path).get(section) or {}
    return bool(cfg.get(key, default))


def resolve_model_repo_id(config_path: Path = CONFIG_PATH) -> tuple[str, str]:
    """Return (repo_id, repo_type) for the trained-model Hub repo."""
    vcfg = load_versioning_config(config_path)
    repo_id = vcfg.get("hf_model_repo_id")
    if not repo_id:
        raise ValueError("versioning.hf_model_repo_id is not set")
    return str(repo_id), str(vcfg.get("hf_model_repo_type", "model"))


def _ensure_repo(repo_id: str, repo_type: str, private: bool) -> HfApi:
    api = HfApi()
    if not api.repo_exists(repo_id=repo_id, repo_type=repo_type):
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
    return api


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

    _ensure_repo(repo_id, repo_type, private)

    result = upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        commit_message=f"Update {path_in_repo} from local {local_dir.name}",
    )
    logger.info("Pushed %s -> %s:%s (%s)", local_dir, repo_id, path_in_repo, result)
    return str(result)


def push_model_tree(
    local_dir: Path | str,
    path_in_repo: str,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    config_path: Path = CONFIG_PATH,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Upload ``local_dir`` under ``path_in_repo`` on the project model repo."""
    default_repo_id, default_repo_type = resolve_model_repo_id(config_path)
    repo_id = repo_id or default_repo_id
    repo_type = repo_type or default_repo_type
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    _ensure_repo(repo_id, repo_type, private)
    result = upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        commit_message=commit_message
        or f"Update {path_in_repo} from local {local_dir.name}",
    )
    logger.info("Pushed %s -> %s:%s (%s)", local_dir, repo_id, path_in_repo, result)
    return str(result)


def push_all_trained_models(
    models_dir: Path | str,
    *,
    config_path: Path = CONFIG_PATH,
    private: bool = False,
) -> list[str]:
    """Upload each trained model dir (rf / xgboost / lightgbm / resnet18)."""
    models_dir = Path(models_dir)
    results: list[str] = []
    for name in TRAINED_MODEL_DIRS:
        local = models_dir / name
        if not local.is_dir():
            raise FileNotFoundError(local)
        results.append(
            push_model_tree(
                local,
                path_in_repo=name,
                config_path=config_path,
                private=private,
                commit_message=f"Update {name} artifacts",
            )
        )
    return results


def push_winner_artifacts(
    models_dir: Path | str,
    *,
    config_path: Path = CONFIG_PATH,
    private: bool = False,
) -> None:
    """Upload ``winner.json`` then ``winner/`` (must run after ``push_all_trained_models``)."""
    models_dir = Path(models_dir)
    winner_json = models_dir / "winner.json"
    winner_dir = models_dir / "winner"
    if not winner_json.is_file():
        raise FileNotFoundError(winner_json)
    if not winner_dir.is_dir():
        raise FileNotFoundError(winner_dir)

    repo_id, repo_type = resolve_model_repo_id(config_path)
    _ensure_repo(repo_id, repo_type, private)
    upload_file(
        path_or_fileobj=str(winner_json),
        path_in_repo="winner.json",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Update winner.json",
    )
    logger.info("Pushed %s -> %s:winner.json", winner_json, repo_id)
    push_model_tree(
        winner_dir,
        path_in_repo="winner",
        repo_id=repo_id,
        repo_type=repo_type,
        config_path=config_path,
        private=private,
        commit_message="Update winner/ artifacts",
    )


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


def pull_winner_artifacts(
    models_dir: Path | str,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> Path:
    """Download ``winner.json`` and ``winner/`` into ``models_dir``."""
    default_repo_id, default_repo_type = resolve_model_repo_id(config_path)
    repo_id = repo_id or default_repo_id
    repo_type = repo_type or default_repo_type
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(models_dir),
        allow_patterns=["winner.json", "winner/**"],
    )
    logger.info("Pulled winner artifacts from %s into %s", repo_id, models_dir)
    return models_dir


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Manual HF versioning (dataset trees + model artifacts)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    push_p = sub.add_parser("push", help="Upload a local tree to the dataset repo")
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

    push_models_p = sub.add_parser(
        "push-models", help="Upload rf/xgboost/lightgbm/resnet18 to the model repo"
    )
    push_models_p.add_argument(
        "models_dir", type=Path, nargs="?", default=ROOT / "models"
    )
    push_models_p.add_argument("--private", action="store_true")

    push_winner_p = sub.add_parser(
        "push-winner", help="Upload winner.json then winner/ to the model repo"
    )
    push_winner_p.add_argument(
        "models_dir", type=Path, nargs="?", default=ROOT / "models"
    )
    push_winner_p.add_argument("--private", action="store_true")

    pull_winner_p = sub.add_parser(
        "pull-winner", help="Download winner.json + winner/ from the model repo"
    )
    pull_winner_p.add_argument(
        "models_dir", type=Path, nargs="?", default=ROOT / "models"
    )
    pull_winner_p.add_argument("--repo-id", default=None)
    pull_winner_p.add_argument("--revision", default=None)

    args = parser.parse_args()
    if args.command == "push":
        push_dataset_tree(
            args.local_dir,
            args.path_in_repo,
            repo_id=args.repo_id,
            private=args.private,
        )
    elif args.command == "pull":
        pull_dataset_tree(
            args.local_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            allow_patterns=args.allow_patterns,
        )
    elif args.command == "push-models":
        push_all_trained_models(args.models_dir, private=args.private)
    elif args.command == "pull-winner":
        pull_winner_artifacts(
            args.models_dir,
            repo_id=args.repo_id,
            revision=args.revision,
        )
    else:
        push_winner_artifacts(args.models_dir, private=args.private)


if __name__ == "__main__":
    main()
