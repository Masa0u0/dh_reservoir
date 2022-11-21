from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

install_requires = [
    "pybind11",
    "numpy",
    "matplotlib",
    "rich",
    "networkx",
]

ext_modules = [
    Pybind11Extension(
        "_encoder",   # PYBIND11_MODULEの名前と一致している必要がある
        sorted(glob("src/encoder/*.cpp")),
    ),
    Pybind11Extension(
        "_liquid_state_machine",
        sorted(glob("src/liquid_state_machine/*.cpp")),
    ),
    Pybind11Extension(
        "_ext_lsm",
        sorted(glob("src/ext_lsm/*.cpp")),
    ),
    Pybind11Extension(
        "_reservoir",
        sorted(glob("src/reservoir/*.cpp")),
    ),
]

setup(
    name="dh_reservoir",
    version="0.0.0",
    author="dohi",
    packages=find_packages(where="lib"),   # lib内の全てをパッケージとみなす(libをルートとする)
    package_dir={"": "lib"},   # ここに共有ファイルが作成される
    install_requires=install_requires,
    ext_modules=ext_modules,
)
