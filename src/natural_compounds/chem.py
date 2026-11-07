"""RDKit helpers shared across corpus preparation and analysis.

All functions accept SMILES strings and return None (or skip) on invalid input
rather than raising, so they can be applied over large, noisy corpora.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str) -> str | None:
    """Return canonical SMILES, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """Bemis-Murcko scaffold as SMILES. Acyclic molecules return an empty string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Morgan (ECFP-like) bit vector as a uint8 numpy array, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def max_tanimoto_to_reference(
    query: Iterable[str], reference: Iterable[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """For each query SMILES, the max Tanimoto similarity to any reference SMILES.

    Used to measure how chemically close a downstream task is to a pretraining corpus.
    Invalid SMILES get NaN. Quadratic in size: subsample the reference for large corpora.
    """
    ref_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        for m in (Chem.MolFromSmiles(s) for s in reference)
        if m is not None
    ]
    out = []
    for s in query:
        mol = Chem.MolFromSmiles(s)
        if mol is None or not ref_fps:
            out.append(np.nan)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)))
    return np.asarray(out, dtype=float)
