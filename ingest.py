import os
from rag_core import load_config, build_or_rebuild_index

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    artifacts = build_or_rebuild_index(
        data_dir=cfg["data_dir"],
        artifacts_dir=cfg["artifacts_dir"],
        embed_model_name=cfg["model"]["embedding"],
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["chunk_overlap"],
        index_name=cfg["index_name"],
        meta_name=cfg["meta_name"],
    )
    print("✅ Index built:")
    for k, v in artifacts.items():
        print(f"  - {k}: {v}")
