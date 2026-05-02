"""Post-generation structural and chemical validator for Chemeleon CIF outputs.

Chemeleon already handles SMACT charge neutrality (pre-generation) and
StructureMatcher deduplication (post-generation). This module validates the
remaining structural, chemical, and physical plausibility checks before
downstream ORB/CHGNet evaluation.

Checks are motivated by failure modes described in the Chemeleon (Park et al.,
Nature Communications 2025) and MatterGen (Zeni et al., Nature 2025) papers.
"""

from __future__ import annotations

import glob
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    formula: str
    space_group: Optional[str]
    space_group_number: Optional[int]
    density: Optional[float]
    energy_ev_per_atom: Optional[float]
    prototype_description: Optional[str]
    failure_reason: Optional[str]
    warnings: list[str] = field(default_factory=list)


class CathodeValidator:
    """Post-generation validator for Chemeleon-generated cathode structures.

    Rules schema (hardcoded defaults):

    # NOW - hardcoded
    rules = {
        "required_elements": ["Li", "O"],
        "banned_elements": ["Tc", "Po", "At", "Rn", "Ra", "Ac", "Pa", "Np", "Pu"],
        "forbidden_spacegroup_numbers": [1],
        "coordination_constraints": {"Li": [4, 6], "Mn": [6], "Co": [6], "Ni": [6], "Fe": [4, 6]},
        "li_fraction_range": [0.05, 0.55],
        "density_range_g_cm3": [1.0, 8.0],
        "energy_threshold_ev_per_atom": 0.5,
        "accepted_prototype_families": None,
        "target_family": "layered_oxide",
        "target_application": "Li-ion cathode",
    }
    """

    def __init__(self, rules: dict) -> None:
        self.rules = rules
        self.target_family = rules.get("target_family")
        self._cnn = CrystalNN()

        # Lazy import to avoid hard failure if chgnet is not installed yet.
        self.chgnet = None
        try:
            from chgnet.model import CHGNet

            self.chgnet = CHGNet.load()
        except Exception as exc:
            logger.warning("CHGNet unavailable, skipping energy check: %s", exc)

        self._condenser = None
        self._describer = None
        try:
            from robocrys import Describer, StructureCondenser

            self._condenser = StructureCondenser()
            self._describer = Describer()
        except Exception as exc:
            logger.warning("Robocrys unavailable, skipping prototype check: %s", exc)

    def validate(self, structure: Structure) -> ValidationResult:
        formula = structure.composition.reduced_formula
        warnings: list[str] = []
        sg_sym = None
        sg_num = None
        density = None
        energy = None
        proto = None

        ok, reason = self._check_interatomic_distance(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason = self._check_cell_angles(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason = self._check_required_elements(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason = self._check_banned_elements(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason, sg_sym, sg_num = self._check_symmetry(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason, warn = self._check_coordination(structure)
        warnings.extend(warn)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason = self._check_stoichiometry(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason, density = self._check_density(structure)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason, energy, warn = self._check_energy(structure)
        warnings.extend(warn)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        ok, reason, proto, warn = self._check_prototype(structure)
        warnings.extend(warn)
        if not ok:
            return self._result(False, reason, formula, sg_sym, sg_num, density, energy, proto, warnings)

        return self._result(True, None, formula, sg_sym, sg_num, density, energy, proto, warnings)

    def _result(
        self,
        passed: bool,
        reason: Optional[str],
        formula: str,
        sg_sym: Optional[str],
        sg_num: Optional[int],
        density: Optional[float],
        energy: Optional[float],
        proto: Optional[str],
        warnings: list[str],
    ) -> ValidationResult:
        return ValidationResult(
            passed=passed,
            formula=formula,
            space_group=sg_sym,
            space_group_number=sg_num,
            density=density,
            energy_ev_per_atom=energy,
            prototype_description=proto,
            failure_reason=reason,
            warnings=warnings,
        )

    def _check_interatomic_distance(self, s: Structure) -> tuple[bool, str]:
        """Chemeleon paper: min distance > 0.5 A and max cell length < 60 A."""
        dm = s.distance_matrix
        np.fill_diagonal(dm, np.inf)
        min_dist = float(np.min(dm))
        if min_dist < 0.5:
            return False, f"Atomic overlap detected: min distance {min_dist:.3f} A < 0.5 A"
        max_len = max(s.lattice.abc)
        if max_len > 60.0:
            return False, f"Cell length {max_len:.2f} A exceeds 60 A limit"
        return True, ""

    def _check_cell_angles(self, s: Structure) -> tuple[bool, str]:
        """Chemeleon paper: degenerate angles indicate collapsed cells."""
        for angle in s.lattice.angles:
            if angle < 10.0 or angle > 170.0:
                return False, f"Degenerate lattice angle: {angle:.1f} deg"
        return True, ""

    def _check_required_elements(self, s: Structure) -> tuple[bool, str]:
        """Chemeleon paper: stochastic generation can drop required elements."""
        required = self.rules.get("required_elements", ["Li", "O"])
        present = {el.symbol for el in s.composition.elements}
        for element in required:
            if element not in present:
                return False, f"Missing required element: {element}"
        return True, ""

    def _check_banned_elements(self, s: Structure) -> tuple[bool, str]:
        """Chemeleon workflow: exclude undesirable elements in cathode targets."""
        banned = set(self.rules.get("banned_elements", []))
        for el in s.composition.elements:
            if el.symbol in banned:
                return False, f"Contains banned element: {el.symbol}"
        return True, ""

    def _check_symmetry(self, s: Structure) -> tuple[bool, str, str, int]:
        """MatterGen/Chemeleon: P1 over-generation is a known failure mode."""
        # symprec=0.1 is the Materials Project standard, not pymatgen default.
        analyzer = SpacegroupAnalyzer(s, symprec=0.1)
        sg_num = int(analyzer.get_space_group_number())
        sg_sym = analyzer.get_space_group_symbol()
        forbidden = self.rules.get("forbidden_spacegroup_numbers", [1])
        if sg_num in forbidden:
            if sg_num == 1:
                return False, "P1 symmetry detected (space group 1) - likely unphysical", sg_sym, sg_num
            return False, f"Forbidden space group: {sg_sym} ({sg_num})", sg_sym, sg_num
        return True, "", sg_sym, sg_num

    def _check_coordination(self, s: Structure) -> tuple[bool, str, list[str]]:
        """MatterGen: reasonable coordination polyhedra for known materials."""
        constraints = self.rules.get("coordination_constraints", {})
        if not constraints:
            return True, "", []
        for idx, site in enumerate(s):
            el = site.specie.symbol
            if el not in constraints:
                continue
            try:
                cn = int(round(self._cnn.get_cn(s, idx)))
            except Exception as exc:
                msg = f"Coordination check skipped for {s.composition.reduced_formula}: {exc}"
                logger.warning(msg)
                return True, "", [msg]
            allowed = constraints[el]
            if cn not in allowed:
                return False, f"{el} site has coordination number {cn}, expected one of {allowed}", []
        return True, "", []

    def _check_stoichiometry(self, s: Structure) -> tuple[bool, str]:
        """Chemeleon: Li fraction sanity for cathode compositions."""
        required = self.rules.get("required_elements", ["Li", "O"])
        if "O" not in {el.symbol for el in s.composition.elements}:
            return False, "No oxygen - not a valid oxide cathode"
        if "Li" not in required:
            return True, ""
        lo, hi = self.rules.get("li_fraction_range", [0.05, 0.55])
        li_frac = float(s.composition.get("Li", 0.0)) / s.composition.num_atoms
        if li_frac < lo or li_frac > hi:
            return False, f"Li fraction {li_frac:.3f} outside expected range [{lo}, {hi}]"
        return True, ""

    def _check_density(self, s: Structure) -> tuple[bool, str, float]:
        """Chemeleon workflow: density range sanity for oxides."""
        lo, hi = self.rules.get("density_range_g_cm3", [1.0, 8.0])
        density = float(s.density)
        if density < lo or density > hi:
            return False, f"Density {density:.2f} g/cm3 outside range [{lo}, {hi}]", density
        return True, "", density

    def _check_energy(self, s: Structure) -> tuple[bool, str, Optional[float], list[str]]:
        """Chemeleon workflow: fast energy screening before expensive DFT."""
        threshold = float(self.rules.get("energy_threshold_ev_per_atom", 0.5))
        if self.chgnet is None:
            msg = "CHGNet not available - energy check skipped"
            logger.warning(msg)
            return True, "", None, [msg]
        try:
            prediction = self.chgnet.predict_structure(s)
            energy = float(prediction["e"])
        except Exception as exc:
            msg = f"CHGNet evaluation failed - energy check skipped: {exc}"
            logger.warning(msg)
            return True, "", None, [msg]
        if energy > threshold:
            return False, f"CHGNet energy {energy:.3f} eV/atom exceeds {threshold}", energy, []
        return True, "", energy, []

    def _check_prototype(self, s: Structure) -> tuple[bool, str, Optional[str], list[str]]:
        """MatterGen: compare against known structural prototypes."""
        accepted = self.rules.get("accepted_prototype_families")
        if self._condenser is None or self._describer is None:
            msg = "Robocrys not available - prototype check skipped"
            logger.warning(msg)
            return True, "", None, [msg]
        try:
            condensed = self._condenser.condense_structure(s)
            desc = self._describer.describe(condensed)
            desc = (desc or "").strip()[:300] or None
        except Exception as exc:
            msg = f"Robocrys failed - prototype check skipped: {exc}"
            logger.warning(msg)
            return True, "", None, [msg]
        if accepted is None or desc is None:
            return True, "", desc, []
        lower = desc.lower()
        if not any(str(token).lower() in lower for token in accepted):
            short = desc[:80] if desc else ""
            return False, f"Prototype '{short}' not in accepted families: {accepted}", desc, []
        return True, "", desc, []


def validate_batch(cif_paths: list[str], rules: dict, verbose: bool = True) -> list[ValidationResult]:
    """Load and validate a batch of CIF files; returns passed results only."""
    validator = CathodeValidator(rules)
    results: list[tuple[str, ValidationResult]] = []

    for path in cif_paths:
        file_path = Path(path)
        try:
            structure = Structure.from_file(file_path)
            result = validator.validate(structure)
            result.warnings = list(result.warnings)
            results.append((file_path.name, result))
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", file_path, exc)

    passed = [r for _, r in results if r.passed]
    if verbose:
        _print_summary(results)
    return passed


def _truncate(text: Optional[str], width: int) -> str:
    if not text:
        return ""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _print_summary(results: list[tuple[str, ValidationResult]]) -> None:
    rows = []
    reasons = Counter()
    for name, r in results:
        if not r.passed and r.failure_reason:
            reasons[r.failure_reason] += 1
        status = "✓ PASS" if r.passed else f"✗ {_truncate(r.failure_reason, 22)}"
        density = f"{r.density:.2f}" if r.density is not None else "-"
        sg = r.space_group or "-"
        rows.append([name, r.formula, sg, density, status])

    headers = ["File", "Formula", "SG", "rho g/cm3", "Result"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    total = sum(widths) + 3 * len(widths) + 1
    print("┌" + "─" * (total - 2) + "┐")
    title = "Chemeleon Output Validation Summary"
    print("│ " + title.ljust(total - 4) + " │")
    print("├" + "─" * (total - 2) + "┤")
    print("│ " + " │ ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " │")
    print("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")
    for row in rows:
        print("│ " + " │ ".join(str(row[i]).ljust(widths[i]) for i in range(len(widths))) + " │")
    print("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")
    passed = sum(r.passed for _, r in results)
    rejected = sum(not r.passed for _, r in results)
    print(f"Passed: {passed} / {len(results)}  |  Rejected: {rejected}")

    if reasons:
        print("Rejection breakdown:")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")


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
    cif_files: list[str] = []
    for directory in search_dirs:
        if os.path.isdir(directory):
            cif_files.extend(glob.glob(os.path.join(directory, "**/*.cif"), recursive=True))
            cif_files.extend(glob.glob(os.path.join(directory, "*.cif")))
    cif_files = list(set(cif_files))

    if not cif_files:
        print("No CIF files found. Run Chemeleon first to generate structures.")
        print("Example: chemeleon sample prompt --text-input 'LiMnO2' --n-samples 10 --n-atoms 8")
    else:
        print(f"Found {len(cif_files)} CIF file(s) to validate.\n")
        survivors = validate_batch(cif_files, rules, verbose=True)
        print(f"\nValidation complete. {len(survivors)} structure(s) passed all checks.")
        if survivors:
            print("\nPassed structures (ready for ORB/CHGNet evaluation):")
            for r in survivors:
                energy = f"{r.energy_ev_per_atom:.3f}" if r.energy_ev_per_atom is not None else "-"
                density = f"{r.density:.2f}" if r.density is not None else "-"
                print(f"  {r.formula} | {r.space_group} | {density} g/cm3 | {energy} eV/atom")
