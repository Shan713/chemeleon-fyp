from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]
    # remove if the line starts with `git+`
    install_requires = [
        line for line in install_requires if not line.startswith("git+")
    ]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="chemeleon",
    version="0.1.3",
    author="Hyunsoo Park",
    author_email="phs68660888@gmail.com",
    description="A Text-guided Crystal Structure Generation Model",
    long_description=long_description,
    packages=find_packages(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "chemeleon=chemeleon.cli:cli",
        ],
    },
)
