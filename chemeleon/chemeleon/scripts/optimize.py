from typing import Optional, Literal
from pathlib import Path
from fire import Fire

from tqdm import tqdm
import numpy as np
from ase import Atoms
from ase.io import read
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

from mace.calculators import mace_mp


def optimize_atoms_list(
    path_dir: str,
    num_optimization: int = 10,
    num_internal_steps: int = 50,
    num_cell_steps: int = 50,
    fmax: float = 0.01,
    cell_relax: bool = True,
    save_dir: Optional[str] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    default_dtype: Literal["float32", "float64"] = "float32",
):
    # set save_dir
    path_dir = Path(path_dir)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = path_dir

    # set filenames
    filenames = list(path_dir.glob("*.cif"))
    # check if the file is already optimized
    already_optimized = list(save_dir.glob("opt_*.cif"))
    already_optimized = [f.name[4:] for f in already_optimized]
    filenames = [f for f in filenames if f.name not in already_optimized]
    # check if files are found
    fail_log_path = path_dir / "fail.log"
    if fail_log_path.exists():
        with open(fail_log_path, "r") as f:
            fail_filenames = f.read().splitlines()
        filenames = [f for f in filenames if str(f) not in fail_filenames]
        print(f"Found {len(fail_filenames)} failed files in {fail_log_path}")
    if len(filenames) == 0:
        raise FileNotFoundError(f"No CIF files found in {path_dir}")
    else:
        print(
            f"Found {len(filenames)} CIF files to optimize in {path_dir}\n"
            f"{len(already_optimized)} files already optimized in {save_dir}\n"
        )

    # set calculator
    calc = mace_mp(device=device, default_dtype=default_dtype)

    # optimize atoms
    for filename in tqdm(filenames):
        print(f"Optimizing {filename} ...")
        atoms = read(filename)
        opt_atoms = optimize_atoms(
            calc=calc,
            atoms=atoms,
            num_optimization=num_optimization,
            num_internal_steps=num_internal_steps,
            num_cell_steps=num_cell_steps,
            fmax=fmax,
            cell_relax=cell_relax,
        )
        if opt_atoms is None:
            print(f"Failed to optimize {filename}")
            # add to fail log
            with open(fail_log_path, "a") as f:
                f.write(f"{filename}\n")
            continue

        # save
        save_path = save_dir / f"opt_{filename.name}"
        opt_atoms.write(str(save_path))


def optimize_atoms(
    calc: Calculator,
    atoms: Atoms,
    num_optimization: int = 50,
    num_internal_steps: int = 50,
    num_cell_steps: int = 50,
    fmax: float = 0.01,
    cell_relax: bool = True,
) -> Optional[Atoms]:
    opt_atoms = atoms.copy()
    convergence = False

    for _ in range(num_optimization):
        opt_atoms = opt_atoms.copy()
        opt_atoms.calc = calc

        # cell relaxation
        if cell_relax:
            filter = FrechetCellFilter(opt_atoms)  # pylint: disable=redefined-builtin
            optimizer = FIRE(filter)
            convergence = optimizer.run(fmax=fmax, steps=num_cell_steps)
            opt_atoms.wrap()
            print(convergence)
            if convergence:
                break

        # internal relaxation
        optimizer = FIRE(opt_atoms)
        convergence = optimizer.run(fmax=fmax, steps=num_internal_steps)
        print(convergence)
        if convergence and not cell_relax:
            break
        # fail if the forces are too large
        forces = filter.get_forces()
        _fmax = np.sqrt((forces**2).sum(axis=1).max())
        if _fmax > 1000:
            return None

    if not convergence:
        return None
    return opt_atoms


if __name__ == "__main__":
    Fire(optimize_atoms_list)
