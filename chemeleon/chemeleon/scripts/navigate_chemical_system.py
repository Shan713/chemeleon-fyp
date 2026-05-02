import itertools
from collections import defaultdict
from pathlib import Path
from fire import Fire
from tqdm import tqdm

from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from smact.screening import smact_validity
from chemeleon.modules.chemeleon import Chemeleon


def navigate_chemical_system(
    elements: list[str] = ["Zn", "Ti", "O"],
    max_stoich: int = 8,
    n_samples: int = 100,
    max_natoms: int = 40,
    max_factor: int = 13,
    save_dir: str = "results/navigate",
):
    model = Chemeleon.load_composition_model()
    model.eval()
    text_targets = model.hparams.text_targets
    print(f"Text targets: {text_targets}")

    # save dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # valid compositions
    all_compositions = [
        Composition(
            {el: amt for el, amt in zip(elements, amt_list)}
        ).reduced_composition
        for amt_list in itertools.product(range(max_stoich + 1), repeat=len(elements))
        if max(amt_list) > 0
    ]
    valid_compositions = [comp for comp in all_compositions if smact_validity(comp)]
    unique_valid_compositions = list(set(valid_compositions))
    print(
        f"Number of unique valid compositions: {len(unique_valid_compositions)} out of {len(all_compositions)}"
    )

    # sampling
    sm = StructureMatcher()
    collections_gen_st = []
    for comp in tqdm(unique_valid_compositions):
        print(f"Sampling for {comp}")
        reduced_natoms = int(comp.num_atoms)
        comp = comp.reduced_composition.alphabetical_formula

        valid_gen_st_list = []
        for f in range(1, max_factor + 1):
            if reduced_natoms * f > max_natoms:
                break
            n_atoms = reduced_natoms * f
            text_input = comp
            print(
                f"Sampling {n_samples} structures for {text_input} with {n_atoms} atoms..."
            )

            # sampling
            gen_atoms_list = model.sample(
                text_input=text_input,
                n_atoms=n_atoms,
                n_samples=n_samples,
            )
            if gen_atoms_list is None:
                continue
            gen_st_list = [
                AseAtomsAdaptor.get_structure(atoms) for atoms in gen_atoms_list
            ]

            # validity check
            for st in gen_st_list:
                if max(st.lattice.abc) > 60:
                    continue
                if st.composition.reduced_composition not in unique_valid_compositions:
                    continue
                valid_gen_st_list.append(st)

        # unique structures
        unique_gen_st_list = [out[0] for out in sm.group_structures(valid_gen_st_list)]
        print(f"Number of unique structures: {len(unique_gen_st_list)}")
        collections_gen_st.extend(unique_gen_st_list)

    # final unique structures
    collections_gen_st = [out[0] for out in sm.group_structures(collections_gen_st)]
    print(f"Number of final unique structures: {len(collections_gen_st)}")

    # save cif files
    idx_list = defaultdict(int)
    for st in collections_gen_st:
        # get composition
        comp = st.composition.reduced_composition.alphabetical_formula.replace(" ", "")
        idx_list[comp] += 1
        # save atoms to cif
        atoms = AseAtomsAdaptor.get_atoms(st)
        filename = f"gen_{comp}_{len(atoms)}_{idx_list[comp]}.cif"
        atoms.write(save_dir / filename)
    print(f"Results saved in {save_dir}")


if __name__ == "__main__":
    Fire(navigate_chemical_system)
