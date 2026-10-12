"""Discrete materials and molecular benchmarks (paper Secs. 5.2-5.3).

Every dataset is a finite candidate set: BEACON chooses which (unseen)
candidate to "evaluate" next and observes its tabulated outcome. Inputs are
min-max normalized; reachability uses 25 equal-width bins over the observed
outcome range for scalar outcomes and a 10 x 10 grid for the joint CO2/CH4
uptake, normalized by the number of regions the full dataset occupies.

Data files live in ``data/materials`` (override with ``BEACON_DATA_DIR``).
Initial designs (``n_init = 10``; 50 for joint gas uptake) were verified
against the tracked plot data in ``results/plot-data``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from beacon.metrics import GridPartition, HistogramPartition, Reachability
from beacon.paths import MATERIALS_DIR
from beacon.problems.base import DiscreteProblem, minmax_normalize

MATERIAL_DEFAULTS = dict(
    n_init=10,
    replicates=20,
    k=10,
    kernel="matern",  # BoTorch 0.10 SingleTaskGP default used by the paper scripts
    likelihood="gamma",
    archive="observed",
    discrete_sampler="auto",
)


def _scalar_problem(name, X, y, n_bins, iterations, description, **default_overrides) -> DiscreteProblem:
    Y = torch.as_tensor(np.array(y, dtype=np.float64)).reshape(-1, 1)  # copy: pandas 3 returns read-only views
    partition = HistogramPartition(float(Y.min()), float(Y.max()), n_bins)
    defaults = dict(MATERIAL_DEFAULTS, iterations=iterations, **default_overrides)
    return DiscreteProblem(
        name,
        X,
        Y,
        metric=Reachability.from_dataset(partition, Y),
        partition=partition,
        defaults=defaults,
        description=description,
    )


def _require(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Set BEACON_DATA_DIR if data live elsewhere.")
    return path


def make_hydrogen(n_bins: int = 25, **_) -> DiscreteProblem:
    """H2 uptake of ~98k MOFs from 7 descriptors (Ghude & Chowdhury, 2023)."""
    with open(_require(MATERIALS_DIR / "hydrogen_input_output.pkl"), "rb") as fh:
        data = pickle.load(fh)
    X = minmax_normalize(torch.as_tensor(np.asarray(data["x"], dtype=np.float64)))
    return _scalar_problem(
        "hydrogen", X, data["y"], n_bins, iterations=80,
        description="Hydrogen uptake capacity of 98,694 MOFs (7 descriptors).",
    )


def make_nitrogen(n_bins: int = 25, **_) -> DiscreteProblem:
    """N2 uptake of 5,224 MOFs from 20 descriptors (Daglar et al.)."""
    df = pd.read_csv(_require(MATERIALS_DIR / "Nitrogen.csv"))
    X = minmax_normalize(torch.as_tensor(df.iloc[:, 1:21].to_numpy(dtype=np.float64)))
    return _scalar_problem(
        "nitrogen", X, df["U_N2 (mol/kg)"].to_numpy(), n_bins, iterations=500,
        description="Nitrogen uptake capacity of 5,224 MOFs (20 descriptors).",
    )


def make_water_solubility(n_bins: int = 25, **_) -> DiscreteProblem:
    """Aqueous solubility (LogS) of 900 molecules from 14 descriptors (Qiao et al.)."""
    df = pd.read_csv(_require(MATERIALS_DIR / "water_set_wide_descriptors.csv"))
    X = minmax_normalize(torch.as_tensor(df.iloc[:, 4:18].to_numpy(dtype=np.float64)))
    return _scalar_problem(
        "water-solubility", X, df.iloc[:, 3].to_numpy(), n_bins, iterations=100,
        description="Water solubility (LogS) of 900 molecules (14 descriptors).",
    )


def load_logd_dataframe(dim: int = 125):
    """Reproduce the paper's LogD descriptor preprocessing; returns ``(X, y)`` arrays."""
    lc_df = pd.read_csv(_require(MATERIALS_DIR / "extract_data_logD.csv"), usecols=["LogD", "Exp_RT"])
    # Remove non-retained molecules.
    index = lc_df[lc_df["Exp_RT"] < 180].index
    lc_df = lc_df.drop(index)
    des_df = pd.read_csv(_require(MATERIALS_DIR / "descriptors_logD.csv"), index_col=0)
    des_df = des_df.drop(des_df.index[index])

    des_with_lc = pd.concat([des_df, lc_df], axis=1)
    feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc["LogD"]) >= 0.90][:-1]
    des_with_lc = des_with_lc.drop(columns=feat_corr)
    # Fill NaNs with column means. (The original per-column `fillna(inplace=True)`
    # is a silent no-op under pandas >= 3 copy-on-write semantics.)
    des_with_lc = des_with_lc.fillna(des_with_lc.mean())
    # Remove all-zero columns.
    des_with_lc = des_with_lc.loc[:, (des_with_lc**2).sum() != 0]
    data = des_with_lc.drop(["LogD"], axis=1)
    # Remove low-variance (<= 0.05) features.
    data_var = data.var()
    data = data.drop(columns=list(data_var[data_var <= 0.05].index))
    # Remove one of each pair of highly correlated (> 0.95) features.
    corr_matrix = data.corr().abs()
    tri_df = corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool)))
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop, axis=1)
    X = data.iloc[:, 1 : (1 + dim)].to_numpy(dtype=np.float64)
    y = des_with_lc["LogD"].to_numpy(dtype=np.float64)
    return X, y


def make_logd(dim: int = 125, n_bins: int = 25, **_) -> DiscreteProblem:
    """Chromatographic LogD of ~2k molecules from 125 descriptors (BEACON-SAAS study)."""
    X, y = load_logd_dataframe(dim)
    X = minmax_normalize(torch.as_tensor(X))
    return _scalar_problem(
        "logd", X, y, n_bins, iterations=100,
        description=f"LogD of {X.shape[0]} molecules ({X.shape[1]} RDKit descriptors).",
    )


def make_esol(n_bins: int = 25, **_) -> DiscreteProblem:
    """ESOL solubility with 2,133-bit ECFP+fragment fingerprints and a Tanimoto GP.

    Requires the optional ``gauche`` package (and RDKit) for featurization.
    """
    try:
        from gauche.dataloader import MolPropLoader
    except ImportError as exc:
        raise ImportError("The ESOL benchmark needs `pip install gauche rdkit` (see README).") from exc
    loader = MolPropLoader()
    loader.load_benchmark("ESOL")
    loader.featurize("ecfp_fragprints")
    X = torch.as_tensor(np.asarray(loader.features, dtype=np.float64))
    return _scalar_problem(
        "esol", X, np.asarray(loader.labels).reshape(-1), n_bins, iterations=300,
        description="ESOL aqueous solubility of 1,128 molecules (fragprint fingerprints, Tanimoto kernel).",
        kernel="tanimoto", likelihood="plain", discrete_sampler="exact",
    )


JOINT_GAS_FEATURES = sorted(
    {
        "D_func-S-1-all", "D_func-S-3-all", "D_func-alpha-1-all", "D_mc_CRY-T-2-all",
        "D_mc_CRY-chi-1-all", "D_mc_CRY-chi-2-all", "D_mc_CRY-chi-3-all", "Df", "Di", "Dif",
        "density [g/cm^3]", "func-S-0-all", "func-alpha-0-all", "func-alpha-1-all",
        "func-alpha-3-all", "func-chi-0-all", "func-chi-2-all", "func-chi-3-all",
        "mc_CRY-S-0-all", "mc_CRY-Z-0-all", "mc_CRY-Z-2-all", "total_POV_gravimetric",
        "total_POV_volumetric", "total_SA_gravimetric", "total_SA_volumetric",
    }
)


def make_joint_gas(n_bins: int = 10, **_) -> DiscreteProblem:
    """Joint CO2 (0.15 bar) / CH4 (5.8 bar) uptake of 7,000 MOFs from 25 descriptors.

    Features are taken in sorted order. (The original script iterated over a
    Python ``set``, whose order changes between interpreter runs, so the
    column order -- and hence the random Fourier features -- was not
    reproducible.)
    """
    df = pd.read_csv(_require(MATERIALS_DIR / "PMOF20K_traindata_7000_train.csv"))
    X = minmax_normalize(torch.as_tensor(df[JOINT_GAS_FEATURES].to_numpy(dtype=np.float64)))
    Y = torch.as_tensor(
        df[["pure_uptake_CO2_298.00_15000", "pure_uptake_methane_298.00_580000"]].to_numpy(dtype=np.float64)
    )
    partition = GridPartition(Y.min(dim=0).values, Y.max(dim=0).values, n_bins)
    defaults = dict(
        MATERIAL_DEFAULTS, n_init=50, iterations=500, kernel="rbf", discrete_sampler="rff"
    )
    return DiscreteProblem(
        "joint-gas",
        X,
        Y,
        metric=Reachability.from_dataset(partition, Y),
        partition=partition,
        defaults=defaults,
        description="Joint CO2/CH4 uptake of 7,000 MOFs (25 descriptors), 10x10 grid.",
    )
