from glob import glob
from setuptools import setup, find_packages

setup(
    name="tune-demo",
    version="0.1.0",
    # about
    author="Thorben Jensen",
    author_email="jensen.thorben@gmail.com",
    license="MIT",
    url="https://github.com/thorbenJensen/tune-demo",
    # source
    packages=find_packages(),
    scripts=glob("bin/*"),
    # dependencies
    install_requires=["scikit-learn"],
    extras_require={
        "dev": ["black", "jupyter", "pylama", "rope"],
        "tune": ["ray", "requests", "pandas", "psutil"],
    },
)
