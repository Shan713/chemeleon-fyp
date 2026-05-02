from typing import List
import numpy as np
from ase import Atoms
from pymatgen.core import Composition

import torch
from torch_geometric.data import Data, Batch


DEFAULT_DTYPE = torch.get_default_dtype()


def atoms_to_pyg_data(
    atoms: Atoms,
    **kwargs,
) -> Data:
    """modified in the mace.data.atomic_data.py"""
    cart_coords = torch.tensor(atoms.positions).to(DEFAULT_DTYPE)
    frac_coords = torch.tensor(atoms.get_scaled_positions()).to(DEFAULT_DTYPE)
    return Data(
        pos=cart_coords,
        atom_types=torch.tensor(atoms.numbers, dtype=torch.long),
        cart_coords=cart_coords,
        frac_coords=frac_coords,
        lattices=torch.tensor(np.array(atoms.cell)).unsqueeze(0).to(DEFAULT_DTYPE),
        natoms=torch.tensor([len(atoms)]),
        **kwargs,
    )


def batch_to_atoms_list(batch: Batch, frac_coords: bool = True) -> List[Atoms]:
    atoms_list = []
    for data in batch.to_data_list():
        atoms = Atoms(
            numbers=data.atom_types.detach().cpu().numpy(),
            cell=data.lattices.squeeze(0).detach().cpu().numpy(),
            pbc=True,
        )
        if frac_coords:
            positions = data.frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
        else:
            positions = data.cart_coords.detach().cpu().numpy()
            atoms.set_positions(positions)
        atoms_list.append(atoms)
    return atoms_list


def convert_reduced_composition(formula: str):
    """
    Convert a formula to a reduced composition in alphabetical order.
    Args:
        formula (str): chemical formula
    Returns:
        str: reduced composition

    Example:
    >>> convert_reduced_composition("Li2O3Mn")
    "Li2 Mn1 O3"
    """

    comp = Composition(formula).alphabetical_formula
    reduced_comp = Composition(comp).reduced_composition
    return str(reduced_comp)
