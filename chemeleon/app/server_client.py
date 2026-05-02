from typing import Optional
import requests
from fire import Fire


def client(
    n_samples: int = 5,
    n_atoms: int = 6,
    text_input: str = "A Crystal Structure of LiMnO4",
    host: str = "127.0.0.1",
    port: int = 8000,
    url: Optional[str] = None,
):
    """
    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_atoms : int
        The number of atoms in the crystal structure.
    text_input : str
        The text input for the model.

    Examples
    --------
    >>> response = client(n_samples=5, n_atoms=6, text_input="A Crystal Structure of LiMnO4", stream=False)
    or
    >>> response = client(n_samples=5, n_atoms=6, text_input="A Crystal Structure of LiMnO4", stream=True)
    >>> for line in response.iter_lines():
    >>>     print(json.loads(line))
    """
    if url is None:
        url = f"http://{host}:{port}/predict"
    else:
        url = f"{url}"
    print(
        f"n_samples: {n_samples}\n"
        f"n_atoms: {n_atoms}\n"
        f"text_input: {text_input}\n"
    )
    response = requests.post(
        url,
        json={
            "n_samples": n_samples,
            "n_atoms": n_atoms,
            "text_input": text_input,
        },
        stream=True,
    )
    return response


if __name__ == "__main__":
    Fire(client)
