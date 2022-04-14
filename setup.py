from setuptools import setup, find_packages


PACKAGE = "pocketseis"

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]

setup(
    name=PACKAGE,
    version="0.1.0",
    description="PocketSeis: a pocket tool for seismology",
    author="Nima Nooshiri",
    author_email="nima@cp.dias.ie; nima.nooshiri@gmail.com",
    classifiers=CLASSIFIERS,
    keywords="Seismology Volcano-Seismology Seismic-Sources",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8')
