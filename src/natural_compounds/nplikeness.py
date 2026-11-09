"""Natural-product-likeness (Ertl et al. 2008), using the model shipped with RDKit."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig


@lru_cache(maxsize=1)
def _scorer():
    sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
    import npscorer

    return npscorer, npscorer.readNPModel()


def np_likeness(smiles: Iterable[str]) -> np.ndarray:
    """Score each molecule, roughly -5 (synthetic) to +5 (natural). NaN if unparsable."""
    npscorer, model = _scorer()
    scores = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        scores.append(np.nan if mol is None else npscorer.scoreMol(mol, model))
    return np.asarray(scores, dtype=float)
