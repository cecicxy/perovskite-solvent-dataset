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

### Purpose

This dataset supports machine learning studies on green solvent discovery for perovskite photovoltaics, including clustering, regression modeling, and similarity analysis to conventional solvents (DMF/DMSO).

## License

Data compiled from public academic sources for research purposes.
