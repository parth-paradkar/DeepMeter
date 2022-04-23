import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

install_requires = [
    "torch==1.9.1"
    "torchvision==0.10.1"
]

tests_require = ["pytest==6.0.1"]

# This call to setup() does all the work
setup(
    name="pysocwatch",
    version="1.0.0",
    description=
    "Python wrapper for Intel SoCWatch",
    author="parth-paradkar",
    author_email="parthparadkar3@gmail.com",
)