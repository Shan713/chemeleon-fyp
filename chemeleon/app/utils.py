import numpy as np
from ase import Atoms
import plotly.graph_objects as go

empty_fig = go.Figure(
    data=[
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
        )
    ],
    layout=go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    ),
)


def atoms_to_dict(atoms):
    """
    Converts an ASE Atoms object to a dictionary for storing.
    """
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
    }


def dict_to_atoms(data):
    """
    Converts a dictionary back to an ASE Atoms object.
    """
    atoms = Atoms(
        symbols=data["symbols"],
        positions=np.array(data["positions"]),
        cell=np.array(data["cell"]),
        pbc=np.array(data["pbc"]),
    )
    return atoms
