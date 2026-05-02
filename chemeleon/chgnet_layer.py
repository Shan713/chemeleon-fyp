"""CHGNet + Materials Project Ehull screening for Chemeleon survivors."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from chgnet.model import CHGNet
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from validator import CathodeValidator


@dataclass
class EnergyResult:
    formula: str
    space_group: str
    chgnet_energy_ev_per_atom: float
    e_above_hull_ev_per_atom: float
    is_metastable: bool
    competing_phases: list[str]


def _load_api_key() -> str | None:
    return os.environ.get("MP_API_KEY")


def _predict_energy(chgnet: CHGNet, structure: Structure) -> float:
    prediction = chgnet.predict_structure(structure)
    return float(prediction["e"])  # eV/atom


def _build_pd(mpr: MPRester, structure: Structure, energy_ev_per_atom: float) -> PhaseDiagram:
    elements = [el.symbol for el in structure.composition.elements]
    entries = mpr.get_entries_in_chemsys(elements)
    total_energy = energy_ev_per_atom * structure.composition.num_atoms
    entries.append(ComputedStructureEntry(structure, total_energy))
    return PhaseDiagram(entries)


def _competing_phases(pd: PhaseDiagram, entry: ComputedStructureEntry) -> list[str]:
    decomp, _ = pd.get_decomp_and_e_above_hull(entry)
    ranked = sorted(decomp.items(), key=lambda item: item[1], reverse=True)
    return [e.composition.reduced_formula for e, _ in ranked[:3]]


def screen_ehull(cif_paths: Iterable[str | Path]) -> list[EnergyResult]:
    api_key = _load_api_key()
    if not api_key:
        print("MP_API_KEY is not set. Export it before running:")
        print("  export MP_API_KEY=your_key_here")
        return []

    chgnet = CHGNet.load()
    results: list[EnergyResult] = []

    with MPRester(api_key) as mpr:
        for path in cif_paths:
            structure = Structure.from_file(path)
            sg = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_symbol()
            energy = _predict_energy(chgnet, structure)
            pd = _build_pd(mpr, structure, energy)
            entry = pd.all_entries[-1]
            ehull = float(pd.get_e_above_hull(entry))
            phases = _competing_phases(pd, entry)
            results.append(
                EnergyResult(
                    formula=structure.composition.reduced_formula,
                    space_group=sg,
                    chgnet_energy_ev_per_atom=energy,
                    e_above_hull_ev_per_atom=ehull,
                    is_metastable=ehull < 0.15,
                    competing_phases=phases,
                )
            )

    return sorted(results, key=lambda r: r.e_above_hull_ev_per_atom)


def _print_table(results: list[EnergyResult]) -> None:
    headers = ["Formula", "SG", "CHGNet e/atom", "Ehull", "Result"]
    rows = []
    for r in results:
        status = "PASS" if r.is_metastable else "FAIL"
        rows.append(
            [
                r.formula,
                r.space_group,
                f"{r.chgnet_energy_ev_per_atom:.3f}",
                f"{r.e_above_hull_ev_per_atom:.3f}",
                status,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = " | "
    print(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(sep.join(str(row[i]).ljust(widths[i]) for i in range(len(widths))))


def _save_json(results: list[EnergyResult]) -> None:
    out_path = Path("results/chgnet_screening.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _collect_cifs(search_dirs: list[str]) -> list[str]:
    cif_files: list[str] = []
    for directory in search_dirs:
        if os.path.isdir(directory):
            cif_files.extend(str(p) for p in Path(directory).rglob("*.cif"))
    return sorted(set(cif_files))


if __name__ == "__main__":
    rules = {
        "required_elements": ["Li", "O"],
        "banned_elements": ["Tc", "Po", "At", "Rn", "Ra", "Ac", "Pa", "Np", "Pu"],
        "forbidden_spacegroup_numbers": [1],
        "coordination_constraints": {
            "Li": [4, 6],
            "Mn": [6],
            "Co": [6],
            "Ni": [6],
            "Fe": [4, 6],
        },
        "li_fraction_range": [0.05, 0.55],
        "density_range_g_cm3": [1.0, 8.0],
        "energy_threshold_ev_per_atom": 0.5,
        "accepted_prototype_families": None,
        "target_family": "layered_oxide",
        "target_application": "Li-ion cathode",
    }

    search_dirs = ["results/prompt", "results/navigate", "results/composition", "results"]
    cif_files = _collect_cifs(search_dirs)
    if not cif_files:
        print("No CIF files found. Run Chemeleon first to generate structures.")
        raise SystemExit(0)

    validator = CathodeValidator(rules)
    survivors: list[str] = []
    for path in cif_files:
        try:
            structure = Structure.from_file(path)
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}")
            continue
        result = validator.validate(structure)
        if result.passed:
            survivors.append(path)

    if not survivors:
        print("No structures passed validation.")
        raise SystemExit(0)

    results = screen_ehull(survivors)
    if not results:
        raise SystemExit(0)
    _print_table(results)
    _save_json(results)
