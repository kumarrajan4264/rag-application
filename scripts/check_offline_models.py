from pathlib import Path

from app.config.settings import EMBEDDING_MODEL, RERANKER_MODEL


def check_model(name: str, path_value: str) -> bool:
    path = Path(path_value)
    exists = path.exists()
    print(f"{name}: {path}")
    print(f"  Exists: {exists}")
    if exists:
        print(f"  Files: {len(list(path.rglob('*')))}")
    return exists


def main():
    print("=" * 70)
    print("OFFLINE MODEL CHECK")
    print("=" * 70)

    embedding_ok = check_model("Embedding model", EMBEDDING_MODEL)
    reranker_ok = check_model("Reranker model", RERANKER_MODEL)

    print()
    if embedding_ok and reranker_ok:
        print("LOCAL MODELS READY")
    else:
        print("LOCAL MODELS NOT READY")
        print()
        print("Copy the downloaded model directories into:")
        print("  models/all-MiniLM-L6-v2")
        print("  models/ms-marco-MiniLM-L-6-v2")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
