import click
from chemeleon.scripts.navigate_chemical_system import navigate_chemical_system
from chemeleon.scripts.sample_target_composition import sample_target_composition
from chemeleon.scripts.sample_prompt import sample_prompt


@click.group(
    help="Chemeleon CLI - A tool for navigating chemical systems and sampling target compositions."
)
def cli():
    pass


### CLI group for navigating chemical systems ###
@cli.group(help="Commands related to chemical system navigation.")
def navigate():
    pass


@navigate.command(
    name="system",
    help="""Navigate a chemical system, e.g., Ti-Zn-O or Li-P-S-Cl.

Examples:

  chemeleon navigate system --elements Zn,Ti,O

  chemeleon navigate system --elements Zn,Ti,O --n-samples 100
""",
)
@click.option(
    "--elements",
    "-e",
    default="Zn,Ti,O",
    show_default=True,
    help="Comma-separated list of elements to navigate the chemical system. e.g. Zn,Ti,O",
)
@click.option(
    "--n-samples",
    default=100,
    show_default=True,
    help="Number of samples to generate.",
)
@click.option(
    "--max-stoich",
    default=8,
    show_default=True,
    help="Maximum stoichiometric factor.",
)
@click.option(
    "--max-natoms",
    default=40,
    show_default=True,
    help="Maximum number of atoms allowed in a structure.",
)
@click.option(
    "--max-factor",
    default=13,
    show_default=True,
    help="Maximum multiplication factor for the composition.",
)
@click.option(
    "--save-dir",
    "-s",
    default="results/navigate",
    show_default=True,
    help="Directory where the generated results will be saved.",
)
def cli_navigate_chemical_system(
    elements: str = "Zn,Ti,O",
    n_samples: int = 100,
    max_stoich: int = 8,
    max_natoms: int = 40,
    max_factor: int = 13,
    save_dir: str = "results/navigate",
):
    elements = elements.split(",")
    navigate_chemical_system(
        elements=elements,
        n_samples=n_samples,
        max_stoich=max_stoich,
        max_natoms=max_natoms,
        max_factor=max_factor,
        save_dir=save_dir,
    )


### CLI group for sampling ###
@cli.group(help="Commands related to sampling target compositions or prompts")
def sample():
    pass


@sample.command(
    name="composition",
    help="""Sample structures with a target composition, e.g., TiO2.

Examples:

  chemeleon sample composition --target-composition TiO2

  chemeleon sample composition -t Li2O --n-samples 50
""",
)
@click.option(
    "--target-composition",
    "-t",
    default="Li2O",
    show_default=True,
    help="Target composition to sample.",
)
@click.option(
    "--n-samples",
    default=100,
    show_default=True,
    help="Number of samples to generate.",
)
@click.option(
    "--max-natoms",
    default=40,
    show_default=True,
    help="Maximum number of atoms allowed in a structure.",
)
@click.option(
    "--max-factor",
    default=13,
    show_default=True,
    help="Maximum multiplication Z factor for the composition.",
)
@click.option(
    "--save-dir",
    "-s",
    default="results/TiO2",
    show_default=True,
    help="Directory where the generated results will be saved.",
)
def cli_sample_target_composition(
    target_composition: str = "TiO2",
    n_samples: int = 100,
    max_natoms: int = 40,
    max_factor: int = 13,
    save_dir: str = "results/TiO2",
):
    sample_target_composition(
        target_composition=target_composition,
        n_samples=n_samples,
        max_natoms=max_natoms,
        max_factor=max_factor,
        save_dir=save_dir,
    )


@sample.command(
    name="prompt",
    help="""Sample structures with a text prompt, e.g., "A Crystal Structure of LiMnO4 with orthorhombic symmetry".

Examples:

  chemeleon sample prompt --text-input "A Crystal Structure of LiMnO4 with orthorhombic symmetry"

  chemeleon sample prompt -t "A Crystal Structure of LiMnO4 with orthorhombic symmetry" --n-samples 50 -n-atoms 6
""",
)
@click.option(
    "--text-input",
    "-t",
    default="A Crystal Structure of LiMnO4 with orthorhombic symmetry",
    show_default=True,
    help="Text input to sample.",
)
@click.option(
    "--n-atoms",
    default=6,
    show_default=True,
    help="Number of atoms in the unit cell.",
)
@click.option(
    "--n-samples",
    default=3,
    show_default=True,
    help="Number of samples to generate.",
)
@click.option(
    "--save-dir",
    "-s",
    default="results/prompt",
)
def cli_sample_prompt(
    text_input: str = "A Crystal Structure of LiMnO4 with orthorhombic symmetry",
    n_atoms: int = 6,
    n_samples: int = 3,
    save_dir: str = "results/prompt",
):
    sample_prompt(
        text_input=text_input,
        n_atoms=n_atoms,
        n_samples=n_samples,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    cli()
