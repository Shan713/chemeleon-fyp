from pathlib import Path
from fire import Fire

from pymatgen.io.ase import AseAtomsAdaptor

from chemeleon.modules.chemeleon import Chemeleon

import wandb


def sample_prompt(
    text_input: str = "A Crystal structure of LiMnO4 with orthorhombic symmetry",
    n_samples: int = 3,
    n_atoms: int = 6,
    save_dir: str = "results/prompt",
):
    model = Chemeleon.load_general_text_model()
    model.eval()
    text_targets = model.hparams.text_targets
    print(f"Text targets: {text_targets}")

    # save dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # sampling
    print(f"Sampling {n_samples} structures for {text_input} with {n_atoms} atoms...")

    # sampling
    gen_atoms_list = model.sample(
        text_input=text_input,
        n_atoms=n_atoms,
        n_samples=n_samples,
    )
    if gen_atoms_list is None:
        print("Sampling failed")
        return
    gen_st_list = [AseAtomsAdaptor.get_structure(atoms) for atoms in gen_atoms_list]

    # save
    for i, st in enumerate(gen_st_list):
        st.to_file(save_dir / f"gen_{i}.cif")
    print(f"Results saved in {save_dir}")


if __name__ == "__main__":
    Fire(sample_prompt)
