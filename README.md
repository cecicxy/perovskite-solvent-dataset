# Perovskite Solar Cell Solvent Dataset

A curated dataset of solvent systems used in perovskite solar cell fabrication, compiled from academic literature.

## Dataset: `perovskite_solvent.csv`

**287 entries** merged from two sources:
- 83 entries with stability data, manually curated from green solvent literature
- 204 entries aggregated from the [Perovskite Database](https://www.perovskitedatabase.com/)

### Columns

| Column | Description |
|--------|-------------|
| `Area (cm2)` | Active cell area in cm² |
| `PCE (%)` | Power conversion efficiency (%) |
| `Stability` | Stability result (e.g., "80% 1000 h"); empty if not reported |
| `Solvent1`–`Solvent4` | Up to 4 solvents used in the perovskite precursor solution |
| `ratio1`–`ratio4` | Molar/volume fraction of each solvent (sums to 1) |

### Common Solvents

DMF, DMSO, NMP, GBL, 2-ME (2-methoxyethanol), ACN, THF, GVL, MAAc, and various green alternatives such as TEP, MAFa, 2-pyrrolidinone, and aniline.

---

## Analysis Scripts

### `mycluster.py`

Clustering utility for solvent molecular descriptors.

- **`mycluster(df, column_name, cluster_method, n_clusters)`**
  Standardizes the selected feature columns (StandardScaler), fits a clustering model, and appends the cluster labels back to the DataFrame. Supported methods: `kmeans`, `dbscan`, `hdbscan`, `gmm`, `birch`, `agglomerative`, `spectral`. Returns the labeled DataFrame and silhouette score.

- **`plot_cluster(...)`**
  Reduces dimensions to 2D via t-SNE or PCA, then plots a scatter chart colored by green/non-green label and styled by cluster. Draws an ellipse around DMF and DMSO as reference points. Saves output as SVG.

### `mydistance.py`

Distance analysis for comparing candidate solvents against a reference solvent.

- **`mydistance(df, column_name, sample_index_col, origin_sample, distance_type)`**
  Standardizes features and computes the distance from every solvent to a reference (e.g., DMF). Supported metrics: `Euclidean`, `Manhattan`, `Cosine`. Returns the DataFrame with a new distance column.

- **`plot_distance(...)`**
  Plots a horizontal gradient bar chart (deep-to-light red) ranking solvents by their distance to the reference. Saves output as SVG.

---

## Purpose

These scripts support machine learning studies on green solvent discovery for perovskite photovoltaics, including clustering, regression modeling, and similarity analysis to conventional solvents (DMF/DMSO).

## License

Data and code compiled from public academic sources for research purposes.
