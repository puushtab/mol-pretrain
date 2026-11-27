"""Mordred descriptors for a fixed list of molecules, computed once and kept.

Mordred is the slow step of every descriptor baseline: 1,613 descriptors per
molecule, single-threaded by default, and several times slower on the large
ring-heavy molecules of a natural-product benchmark than on drug-like ones. A
serial pass over the 60,170 origin-benchmark molecules did not finish in three
hours.

So the matrix is computed in parallel, in shards that each land on disk as they
finish, and the assembled raw matrix is kept under

    <data>/processed/benchmark_descriptors/mordred__<key>/
        raw.npy            float64, molecules x descriptors, NaN where undefined
        descriptors.json   the descriptor names, in column order
        smiles.txt         the molecules, in row order
        shards/            removed once raw.npy is written

`<key>` is a digest of the molecule list in order, the same one the probe keys
its feature cache on, so any task over the same molecules finds the same store.
The matrix is RAW: no column dropped, nothing imputed, nothing standardised.
Whatever a consumer needs (the probe's median fill, Chemprop's own scaling) it
applies itself, so one store serves all of them.

A shard is deterministic in its molecules, so a job that is preempted or times
out resumes where it stopped and the result does not depend on how many workers
ran or in what order they finished. An unparseable molecule is a row of zeros,
which is what the serial code this replaces wrote.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import time
from pathlib import Path

import numpy as np

from natural_compounds.paths import DATA_PROCESSED

SHARD_SIZE = 250

_calculator = None


def molecule_key(smiles: list[str]) -> str:
    """Digest of a molecule list in order. Identical to the probe's cache key."""
    digest = hashlib.sha256()
    for s in smiles:
        digest.update(s.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def store_dir(smiles: list[str]) -> Path:
    return DATA_PROCESSED / "benchmark_descriptors" / f"mordred__{molecule_key(smiles)}"


def default_nproc() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def _calculator_here():
    global _calculator
    if _calculator is None:
        from mordred import Calculator, descriptors
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        _calculator = Calculator(descriptors, ignore_3D=True)
    return _calculator


def descriptor_names() -> list[str]:
    return [str(d) for d in _calculator_here().descriptors]


def _describe(block: list[str]) -> np.ndarray:
    from rdkit import Chem

    calculator = _calculator_here()
    out = np.zeros((len(block), len(calculator.descriptors)), dtype=np.float64)
    for i, s in enumerate(block):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[i] = np.asarray(calculator(mol), dtype=np.float64)
    return out


def _shard(job: tuple[int, list[str], str]) -> tuple[int, float]:
    index, block, directory = job
    started = time.time()
    target = Path(directory) / f"shard_{index:05d}.npy"
    scratch = target.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(scratch, _describe(block))
    os.replace(scratch, target)
    return index, time.time() - started


def mordred_matrix(smiles: list[str], nproc: int | None = None, log=None) -> np.ndarray:
    """The raw Mordred matrix for `smiles`, from the store if it is there."""
    log = log or (lambda message: print(message, flush=True))
    directory = store_dir(smiles)
    raw = directory / "raw.npy"
    if raw.exists():
        matrix = np.load(raw)
        if matrix.shape[0] != len(smiles):
            raise SystemExit(f"{raw} has {matrix.shape[0]} rows for {len(smiles)} molecules")
        return matrix

    nproc = nproc or default_nproc()
    shards = directory / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    blocks = [smiles[i : i + SHARD_SIZE] for i in range(0, len(smiles), SHARD_SIZE)]
    todo = [
        (i, block, str(shards))
        for i, block in enumerate(blocks)
        if not (shards / f"shard_{i:05d}.npy").exists()
    ]
    log(
        f"   mordred: {len(smiles):,} molecules, {len(blocks)} shards, "
        f"{len(blocks) - len(todo)} already on disk, {nproc} workers"
    )
    started = time.time()
    if todo:
        if nproc > 1:
            # Longest SMILES first: the slow molecules are the big ones, and a
            # worker that starts one last leaves the others idle at the end.
            todo.sort(key=lambda job: -sum(len(s) for s in job[1]))
            with multiprocessing.get_context("spawn").Pool(nproc) as pool:
                for done, (_index, seconds) in enumerate(pool.imap_unordered(_shard, todo), 1):
                    if done % 20 == 0 or done == len(todo):
                        log(
                            f"   mordred: {done}/{len(todo)} shards, "
                            f"{(time.time() - started) / 60:.1f} min, last took {seconds:.0f}s"
                        )
        else:
            for job in todo:
                _shard(job)

    matrix = np.concatenate([np.load(shards / f"shard_{i:05d}.npy") for i in range(len(blocks))])
    if matrix.shape[0] != len(smiles):
        raise SystemExit(f"assembled {matrix.shape[0]} rows for {len(smiles)} molecules")
    scratch = directory / f"raw.{os.getpid()}.tmp.npy"
    np.save(scratch, matrix)
    os.replace(scratch, raw)
    (directory / "descriptors.json").write_text(json.dumps(descriptor_names(), indent=0))
    (directory / "smiles.txt").write_text("\n".join(smiles) + "\n")
    shutil.rmtree(shards)
    log(f"   mordred: wrote {raw} {matrix.shape}")
    return matrix
