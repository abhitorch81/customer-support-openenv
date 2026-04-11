from __future__ import annotations

import copy
from typing import Any

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()


def _require_rdkit():  # noqa: ANN202
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem, Descriptors, Lipinski, QED  # noqa: F401
        from rdkit.Chem import rdMolDescriptors  # noqa: F401
        from rdkit import DataStructs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RDKit is required: pip install rdkit") from exc


def mol_from_smiles(smiles: str) -> Any | None:
    _require_rdkit()
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def smiles_from_mol(mol: Any) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol)


def morgan_fp(mol: Any, radius: int = 2, n_bits: int = 2048) -> Any:
    from rdkit.Chem import AllChem

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_to_reference(mol: Any, ref_smiles: str) -> float:
    ref = mol_from_smiles(ref_smiles)
    if ref is None or mol is None:
        return 0.0
    from rdkit import DataStructs

    fp1 = morgan_fp(mol)
    fp2 = morgan_fp(ref)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def sa_score(mol: Any) -> float:
    """Synthetic accessibility (lower = easier); fallback if Contrib missing."""
    try:
        from rdkit.Chem import RDConfig
        import importlib.util
        import os

        path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("sascorer", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return float(mod.calculateScore(mol))
    except Exception:
        pass
    from rdkit.Chem import rdMolDescriptors

    nrb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return min(10.0, 2.0 + 0.45 * nrb)


# Small PAINS-like alert subset (proxy toxicity / reactivity risk)
_PAINS_SMARTS = [
    "c1cncnc1",  # diazine patterns (simplified)
    "[N+](=O)[O-]",  # nitro
    "N=N",  # azo
    "C(=O)N(O)",  # hydroxamic acid-like
    "S(=O)(=O)Cl",  # sulfonyl chloride
]


def pains_alert_count(mol: Any) -> int:
    _require_rdkit()
    from rdkit import Chem

    n = 0
    for smarts in _PAINS_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            n += 1
    return n


def lipinski_violations(mol: Any, max_mw: float = 500, max_logp: float = 5) -> int:
    from rdkit.Chem import Descriptors, Lipinski

    violations = 0
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    if mw > max_mw:
        violations += 1
    if logp > max_logp:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def compute_descriptor_bundle(mol: Any, ref_smiles: str) -> dict[str, float]:
    from rdkit.Chem import Descriptors, Lipinski, QED

    qed = float(QED.qed(mol))
    sa = sa_score(mol)
    pains = float(pains_alert_count(mol))
    sim = tanimoto_to_reference(mol, ref_smiles)
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "qed": qed,
        "sa_score": sa,
        "toxicity_proxy": pains,
        "affinity_proxy": sim,
    }


def format_descriptor_block(d: dict[str, float]) -> str:
    return (
        f"MW: {d['mw']:.1f} | LogP: {d['logp']:.2f} | HBD: {int(d['hbd'])} | HBA: {int(d['hba'])}\n"
        f"TPSA: {d['tpsa']:.1f} | QED: {d['qed']:.2f} | SA: {d['sa_score']:.2f}\n"
        f"PAINS-like alerts: {int(d['toxicity_proxy'])} | affinity_proxy (Tanimoto to ref): {d['affinity_proxy']:.3f}"
    )


def replace_substructure(smiles: str, query_smarts: str, replacement_smiles: str) -> str | None:
    _require_rdkit()
    from rdkit import Chem

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    pat = Chem.MolFromSmarts(query_smarts)
    repl = Chem.MolFromSmiles(replacement_smiles)
    if pat is None or repl is None:
        return None
    if not mol.HasSubstructMatch(pat):
        return None
    try:
        products = Chem.ReplaceSubstructs(mol, pat, repl)
    except Exception:
        return None
    if not products:
        return None
    out = products[0]
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return smiles_from_mol(out)


def remove_substructure(smiles: str, query_smarts: str) -> str | None:
    _require_rdkit()
    from rdkit import Chem

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    pat = Chem.MolFromSmarts(query_smarts)
    if pat is None or not mol.HasSubstructMatch(pat):
        return None
    try:
        out = Chem.DeleteSubstructs(mol, pat)
    except Exception:
        return None
    if out.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return smiles_from_mol(out)


# Canned reaction SMARTS for MVP "add_group" / bioisostere
_CANNED_RXN: dict[str, str] = {
    "methyl_aromatic": "[cH1:1]>>[c:1]C",
    "fluoro_aromatic": "[cH1:1]>>[c:1]F",
    "hydroxyl_to_fluoro": "[OH:1]>>[F:1]",
    "nitro_to_amino": "[N+](=O)[O-]>>N",
}


def run_canned_reaction(smiles: str, key: str) -> str | None:
    _require_rdkit()
    from rdkit.Chem import AllChem

    rxn_smarts = _CANNED_RXN.get(key)
    if not rxn_smarts:
        return None
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        ps = rxn.RunReactants((mol,))
    except Exception:
        return None
    if not ps:
        return None
    try:
        p0 = ps[0][0]
        from rdkit import Chem

        Chem.SanitizeMol(p0)
        return smiles_from_mol(p0)
    except Exception:
        return None


def scaffold_murcko_smiles(mol: Any) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    core = MurckoScaffold.GetScaffoldForMol(mol)
    return smiles_from_mol(core) if core else ""


def constraint_penalty(
    d: dict[str, float],
    max_mw: float,
    logp_min: float,
    logp_max: float,
    max_tpsa: float | None,
    min_qed: float,
) -> float:
    pen = 0.0
    if d["mw"] > max_mw:
        pen += min(2.0, (d["mw"] - max_mw) / 100.0)
    if d["logp"] < logp_min:
        pen += (logp_min - d["logp"]) * 0.3
    if d["logp"] > logp_max:
        pen += (d["logp"] - logp_max) * 0.3
    if max_tpsa is not None and d["tpsa"] > max_tpsa:
        pen += min(1.5, (d["tpsa"] - max_tpsa) / 80.0)
    if d["qed"] < min_qed:
        pen += (min_qed - d["qed"]) * 0.8
    return pen


def composite_reward(
    d: dict[str, float],
    task: dict[str, Any],
    prev_d: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Return step reward and component breakdown (for logging)."""
    tp = task.get("target_profile", {})
    w = task.get("reward_weights", {})
    w1 = float(w.get("affinity", 1.2))
    w2 = float(w.get("qed", 0.9))
    w3 = float(w.get("toxicity", 0.7))
    w4 = float(w.get("sa", 0.15))
    w5 = float(w.get("constraints", 1.0))

    max_mw = float(tp.get("max_mw", 500))
    logp_min = float(tp.get("logp_min", 0.5))
    logp_max = float(tp.get("logp_max", 5.0))
    max_tpsa_raw = tp.get("max_tpsa")
    max_tpsa_arg = float(max_tpsa_raw) if max_tpsa_raw is not None else None
    min_qed = float(tp.get("min_qed", 0.2))

    cpen = constraint_penalty(d, max_mw, logp_min, logp_max, max_tpsa_arg, min_qed)

    # affinity_proxy already 0..1 (Tanimoto)
    aff = d["affinity_proxy"]
    qed = d["qed"]
    tox = d["toxicity_proxy"]
    sa = d["sa_score"]

    base = w1 * aff + w2 * qed - w3 * tox - w4 * (sa / 10.0) - w5 * cpen

    # delta reward if we have previous
    delta = 0.0
    if prev_d is not None:
        delta = 0.35 * (aff - prev_d["affinity_proxy"])
        delta += 0.25 * (qed - prev_d["qed"])
        delta -= 0.2 * max(0, tox - prev_d["toxicity_proxy"])

    reward = 0.15 * base + delta
    components = {
        "affinity_term": aff,
        "qed_term": qed,
        "toxicity_term": tox,
        "sa_term": sa / 10.0,
        "constraint_penalty": cpen,
        "base": base,
    }
    return round(reward, 4), components
