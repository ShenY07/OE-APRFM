"""Sets up kipack."""

import os
from io import open

from setuptools import find_packages, setup

version = "0.1.0"

project_name = "APRFM"

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _get_requirements():
    """Parses requirements.txt file."""
    install_requires_tmp = []
    dependency_links_tmp = []
    with open(os.path.join(os.path.dirname(__file__), "./requirements.txt"), "r") as f:
        for line in f:
            package_name = line.strip()
            if package_name.startswith("-e "):
                dependency_links_tmp.append(package_name[3:].strip())
            else:
                install_requires_tmp.append(package_name)
    return install_requires_tmp, dependency_links_tmp


install_requires, dependency_links = _get_requirements()


print("install_requires: ", install_requires)
print("dependency_links: ", dependency_links)


setup(
    name=project_name,
    version=version,
    description="A jax-based solver for kinetic equations",
    long_description=long_description,
    long_description_content_type="markdown",
    author="Keke Wu",
    author_email="wukekever@sjtu.edu.cn",
    url="https://github.com/wukekever/aprfm",
    license="GPL-3.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={project_name: ["*.txt"]},
    install_requires=install_requires,
    python_requires=">=3.10",
)
