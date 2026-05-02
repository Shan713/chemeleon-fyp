import os
import requests


def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL to a local destination."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP errors
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Ensure directory exists
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
