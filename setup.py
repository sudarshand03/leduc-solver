from setuptools import setup, find_packages

setup(
    name="leduc_solver",               
    version="0.1.0",                   
    description="Leduc Poker CFR solver",
    author="Sudarshan Damodharan",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        "rlcard",
        "numpy",
        "torch",
    ],
)
