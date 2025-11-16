from setuptools import setup, find_packages

setup(
    name="StanLogic",
    version="1.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Stan's Technologies",
    description="An advanced KMap solver and logic simplification engine",
    python_requires=">=3.8",
)
