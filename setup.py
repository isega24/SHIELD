import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SHIELD_isega24",
    version="0.0.1",
    author="Iván Sevillano Garcia",
    author_email="isevillano@ugr.es",
    description="Selective Hidden Input Evaluation for Learning Dynamics",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3"],
    packages=["SHIELD", "SHIELD.SHIELD", "SHIELD.procedures"],
    install_requires=[
        "torch",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "torchvision",
        "efficientnet_pytorch",
        "tqdm",
        "opencv-python",
        "IPython",
        "seaborn",
        "plotly",
        "ipywidgets",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinxcontrib.bibtex",
        "nbsphinx",
        "wget",
        "pandoc",
        "ReVel",
        "torchsummary",
        "tensorboard",
    ],
)