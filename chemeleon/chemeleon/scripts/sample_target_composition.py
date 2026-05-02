from collections import defaultdict
from pathlib import Path
from fire import Fire

from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from chemeleon.modules.chemeleon import Chemeleon


def sample_target_composition(
    target_composition: str = "TiO2",
    n_samples: int = 100,
    max_natoms: int = 40,
    max_factor: int = 13,
    save_dir: str = "results/TiO2/0_gen_atoms",
):
    model = Chemeleon.load_composition_model()
    model.eval()
    text_targets = model.hparams.text_targets
    print(f"Text targets: {text_targets}")

    # save dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # valid compositions
    reduced_comp = Composition(target_composition).reduced_composition
    reduced_natoms = int(reduced_comp.num_atoms)
    comp = reduced_comp.alphabetical_formula
    print(f"target composition: {comp}")

    # sampling
    sm = StructureMatcher(angle_tol=10)
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
        gen_st_list = [AseAtomsAdaptor.get_structure(atoms) for atoms in gen_atoms_list]

        # validity check
        for st in gen_st_list:
            if max(st.lattice.abc) > 60:
                continue
            if st.composition.reduced_composition.alphabetical_formula != comp:
                continue
            valid_gen_st_list.append(st)
        print(len(valid_gen_st_list))

    # unique structures
    collections_gen_st = [out[0] for out in sm.group_structures(valid_gen_st_list)]
    print(f"Number of unique structures: {len(collections_gen_st)}")

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
    Fire(sample_target_composition)
