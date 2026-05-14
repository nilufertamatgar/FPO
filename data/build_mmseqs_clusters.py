#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from pathlib import Path
import pickle
from typing import Optional


def load_mmseqs_tsv(path: Path) -> OrderedDict:
    clusters = OrderedDict()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            clusters.setdefault(rep, []).append(member)
    return clusters


def build_clusters(
    train_tsv: Path,
    valid_tsv: Path,
    out_dir: Path,
    test_tsv: Optional[Path] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_to_pdb_chain = {}
    pdb_chain_to_cluster = {}
    train_cluster_ids = []
    valid_cluster_ids = []
    test_cluster_ids = []

    next_cluster_id = 0
    overlap = set()

    split_inputs = [
        ("train", train_tsv, train_cluster_ids),
        ("valid", valid_tsv, valid_cluster_ids),
    ]
    include_test = False
    if test_tsv is not None:
        if test_tsv.exists():
            split_inputs.append(("test", test_tsv, test_cluster_ids))
            include_test = True
        else:
            print(f"Warning: test TSV not found: {test_tsv}. Skipping test split.")

    for split_name, tsv_path, cluster_ids in split_inputs:
        clusters = load_mmseqs_tsv(tsv_path)
        for _, members in clusters.items():
            cluster_id = next_cluster_id
            cluster_to_pdb_chain[cluster_id] = members
            for m in members:
                if m in pdb_chain_to_cluster:
                    overlap.add(m)
                pdb_chain_to_cluster[m] = cluster_id
            cluster_ids.append(cluster_id)
            next_cluster_id += 1

    with open(out_dir / "cluster_to_pdb_chain.pkl", "wb") as f:
        pickle.dump(cluster_to_pdb_chain, f)
    with open(out_dir / "pdb_chain_to_cluster.pkl", "wb") as f:
        pickle.dump(pdb_chain_to_cluster, f)

    with open(out_dir / "train_clusters.txt", "w") as f:
        for cid in train_cluster_ids:
            f.write(f"{cid}\n")
    with open(out_dir / "valid_clusters.txt", "w") as f:
        for cid in valid_cluster_ids:
            f.write(f"{cid}\n")
    if include_test:
        with open(out_dir / "test_clusters.txt", "w") as f:
            for cid in test_cluster_ids:
                f.write(f"{cid}\n")

    summary_path = out_dir / "cluster_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"train_clusters\t{len(train_cluster_ids)}\n")
        f.write(f"valid_clusters\t{len(valid_cluster_ids)}\n")
        if include_test:
            f.write(f"test_clusters\t{len(test_cluster_ids)}\n")
        f.write(f"total_clusters\t{len(cluster_to_pdb_chain)}\n")
        f.write(f"total_members\t{len(pdb_chain_to_cluster)}\n")
        f.write(f"overlap_members\t{len(overlap)}\n")

    return summary_path, overlap


def main():
    parser = argparse.ArgumentParser(
        description="Convert mmseqs cluster TSVs into train/valid/test cluster files."
    )
    parser.add_argument(
        "--train_tsv",
        type=Path,
        default=Path("/dapustor/nilufer/ADFLIP/data/cluster/mmseqs_train/train_clusters.tsv"),
        help="Path to train mmseqs clusters TSV.",
    )
    parser.add_argument(
        "--valid_tsv",
        type=Path,
        default=Path("/dapustor/nilufer/ADFLIP/data/cluster/mmseqs_valid/valid_clusters.tsv"),
        help="Path to valid mmseqs clusters TSV.",
    )
    parser.add_argument(
        "--test_tsv",
        type=Path,
        default=Path("/dapustor/nilufer/ADFLIP/data/cluster/mmseqs_test/test_clusters.tsv"),
        help="Path to test mmseqs clusters TSV (optional).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/dapustor/nilufer/ADFLIP/data/new_cluster"),
        help="Output directory for cluster files.",
    )
    args = parser.parse_args()

    summary_path, overlap = build_clusters(
        args.train_tsv,
        args.valid_tsv,
        args.out_dir,
        test_tsv=args.test_tsv,
    )
    print(f"Cluster files written to: {args.out_dir}")
    print(f"Summary: {summary_path}")
    if overlap:
        print(f"Warning: {len(overlap)} overlapping chain IDs between train/valid.")


if __name__ == "__main__":
    main()
