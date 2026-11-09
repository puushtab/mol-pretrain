"""RDKit helpers for corpus preparation and analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

PROPERTIES = {
    "mol_weight": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "rings": Descriptors.RingCount,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "fraction_csp3": Descriptors.FractionCSP3,
}


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def is_multifragment(smiles: str) -> bool:
    return "." in smiles


def murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _morgan.GetFingerprintAsNumPy(mol).astype(np.uint8)


def fingerprint_matrix(smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprints of the valid molecules, and a mask saying which inputs they came from."""
    rows, valid = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        valid.append(mol is not None)
        if mol is not None:
            rows.append(_morgan.GetFingerprintAsNumPy(mol))
    matrix = np.asarray(rows, dtype=np.uint8) if rows else np.zeros((0, N_BITS), np.uint8)
    return matrix, np.asarray(valid, dtype=bool)


def tanimoto_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between two fingerprint matrices.

    The intersection is a matmul, which BLAS does far faster than a Python loop
    over RDKit calls. Rows are processed in blocks so the peak allocation stays at
    one block rather than the whole output.
    """
    ref = reference.astype(np.float32)
    ref_bits = ref.sum(axis=1)
    out = np.empty((len(query), len(reference)), dtype=np.float32)
    for start in range(0, len(query), 1024):
        block = query[start : start + 1024].astype(np.float32)
        intersection = block @ ref.T
        union = block.sum(axis=1)[:, None] + ref_bits - intersection
        out[start : start + 1024] = intersection / np.maximum(union, 1.0)
    return out


def max_tanimoto_to_reference(query: Iterable[str], reference: Iterable[str]) -> np.ndarray:
    """For each query molecule, its similarity to the closest reference molecule."""
    query = list(query)
    ref_fps, _ = fingerprint_matrix(reference)
    query_fps, valid = fingerprint_matrix(query)

    out = np.full(len(query), np.nan)
    if len(ref_fps) and len(query_fps):
        out[valid] = tanimoto_matrix(query_fps, ref_fps).max(axis=1)
    return out


def descriptive_stats(smiles: Sequence[str]) -> dict[str, float]:
    """Mean and standard deviation of a few cheap properties."""
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]
    stats: dict[str, float] = {"n": float(len(mols))}
    for name, prop in PROPERTIES.items():
        values = np.array([prop(m) for m in mols], dtype=float)
        stats[f"{name}_mean"] = float(values.mean())
        stats[f"{name}_std"] = float(values.std(ddof=1))
    return stats
