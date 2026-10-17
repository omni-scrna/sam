#!/usr/bin/env python3
"""FEAT module: Self-Assembling Manifold gene selection (atarashansky/self-assembling-manifold).

SAM alternates a kNN graph on the cells with a gene weight: each gene's
dispersion after averaging over kNN neighbourhoods, so genes that vary
smoothly along the manifold rise. The graph is rebuilt on the reweighted genes
until the weights converge. Genes are ranked by the final weight and the top
--number_selected are kept.

SAM runs its own preprocessing (cell-median library size, log2(1+x), zero out
values <= 1, drop genes expressed in > --thresh_high of cells) on raw counts from
rawdata_h5ad layers["counts"], subset to the cells and genes of --normalized_h5
(the fe-scanpy pearson_residuals convention), so FILT's filter applies and the
NORM arm does not change which genes are chosen. The written values are
normalized_h5 restricted to those genes.
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy import sparse
from samalg import SAM
from samalg import utilities as ut

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common)
from common import cli  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="FEAT module: SAM")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "FEAT")     # the FEAT stage I/O, from the plan's schema
    p.add_argument("--number_selected", type=int, required=True,
                   help="number of top-weighted genes to keep")
    p.add_argument("--random_seed", type=int, default=0,
                   help="seed for SAM's PCA and the hnswlib kNN index")
    p.add_argument("--thresh_high", type=float, default=0.96,
                   help="drop genes detected in more than this fraction of cells "
                        "(upstream default 0.96; 1.0 keeps all)")
    return p.parse_args()


def read_tenx(path):
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][:])
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        genes = g["genes"][:].astype(str)
        cells = g["barcodes"][:].astype(str)
    return m, genes, cells  # genes x cells


def write_tenx(path, m, genes, cells):
    m = sparse.csc_matrix(m)
    m.sort_indices()
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"] = m.data
        g["indices"] = m.indices.astype(np.uint32)
        g["indptr"] = m.indptr.astype(np.int64)
        g["shape"] = np.array(m.shape, dtype=np.uint32)
        g["genes"] = np.array(genes, dtype="S")
        g["barcodes"] = np.array(cells, dtype="S")


def load_counts(rawdata_h5ad, cells, genes):
    """rawdata layers["counts"] subset to (cells, genes) -> cells x genes AnnData."""
    raw = ad.read_h5ad(rawdata_h5ad)
    if "counts" not in raw.layers:
        raise ValueError("SAM needs raw counts in rawdata_h5ad layers['counts']")
    x = raw[cells, genes].layers["counts"]
    return ad.AnnData(X=sparse.csr_matrix(x, dtype=np.float32),
                      obs={"obs_names": cells}, var={"var_names": genes})


def hnsw_single_thread(x, ef=200, M=48, n_neighbors=100, seed=100):
    """samalg.utilities.nearest_neighbors_hnsw with num_threads=1.

    Upstream builds the index on every core; the insertion order, hence the
    graph and the gene weights, then change between same-seed runs. One thread
    makes it deterministic; --random_seed replaces hnswlib's fixed level seed.
    """
    import hnswlib
    p = hnswlib.Index(space="cosine", dim=x.shape[1])
    p.init_index(max_elements=x.shape[0], ef_construction=ef, M=M, random_seed=seed)
    p.add_items(x, np.arange(x.shape[0]), num_threads=1)
    p.set_ef(ef)
    return p.knn_query(x, k=n_neighbors, num_threads=1)


def rank_genes(counts, seed, thresh_high):
    """counts: cells x genes AnnData of raw counts -> all genes, best SAM weight first."""
    # calc_nnm looks nearest_neighbors_hnsw up at call time
    ut.nearest_neighbors_hnsw = partial(hnsw_single_thread, seed=seed)
    sam = SAM(counts=counts)
    sam.preprocess_data(thresh_high=thresh_high)
    sam.run(projection=None, seed=seed)  # upstream defaults: k=20, npcs=150, n_genes=3000
    w = sam.adata.var["weights"]
    print(f"SAM: {(w > 0).sum()} / {len(w)} genes with nonzero weight")
    return list(sam.adata.uns["ranked_genes"])


def main():
    args = parse_args()
    norm, genes, cells = read_tenx(args.normalized_h5)
    print(f"normalized_h5 (genes x cells): {norm.shape}")
    if not 0 < args.number_selected <= len(genes):
        raise ValueError(f"--number_selected must be in (0, {len(genes)}], got {args.number_selected}")
    if not 0 < args.thresh_high <= 1:
        raise ValueError(f"--thresh_high must be in (0, 1], got {args.thresh_high}")

    selected = set(rank_genes(load_counts(args.rawdata_h5ad, cells, genes),
                              args.random_seed, args.thresh_high)
                   [:args.number_selected])
    keep = np.array([g in selected for g in genes])
    assert keep.sum() == args.number_selected
    print(f"selected {keep.sum()} / {len(genes)} genes")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tenx(out / f"{args.name}_normalized_selected.h5", norm[keep, :], genes[keep], cells)


if __name__ == "__main__":
    main()
