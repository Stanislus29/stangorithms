from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="StanLogic",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Stan's Technologies",
    description="An advanced KMap solver and logic simplification engine",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "web": ["Flask>=2.0", "Flask-Cors>=3.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
)
