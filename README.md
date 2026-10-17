# sam

FEAT stage arm: Self-Assembling Manifold gene selection with
[atarashansky/self-assembling-manifold](https://github.com/atarashansky/self-assembling-manifold)
(`sc-sam` 2.0.2, Tarashansky et al., eLife 2019).

SAM reweights genes by their kNN-averaged dispersion and rebuilds the kNN graph
on the reweighted genes until the weights converge. The top `--number_selected`
genes by final weight are kept. It fits on raw counts (`rawdata_h5ad`
`layers["counts"]`, subset to the cells and genes of `normalized_h5`) with SAM's
own preprocessing, and writes `normalized_h5` restricted to the selected genes.

Upstream builds its hnswlib index on all cores, which changes the gene set
between same-seed runs. The entrypoint swaps in a single-threaded build seeded
with `--random_seed`, so a seed gives the same genes every run.

```bash
pixi run check
pixi run python feat-select.py --output_dir out --name <dataset> \
  --rawdata_h5ad <d>.h5ad --normalized_h5 <d>_normalized.h5 \
  --filtered_cellids <d>_cellids.txt.gz --filtered_featureids <d>_featureids.txt.gz \
  --properties_info <d>_properties.yaml --number_selected 2000 --random_seed 0
```

duo-zhengmix8eq (3993 cells, 11998 genes): 10 s, 0.85 GB peak RSS.
