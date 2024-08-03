import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-ldl",
    version="0.0.3",
    author="SpriteMisaka",
    author_email="SpriteMisaka@gmail.com",
    description="Label distribution learning (LDL) and label enhancement (LE) toolkit implemented in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpriteMisaka/PyLDL",

    packages=setuptools.find_packages(),
    package_data={
        '': ['*.m']
    },
    include_package_data=True,
    exclude_package_data={
        '': ['__pycache__', 'LDLPackage_v1.2']
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        "matplotlib",
        "numpy",
        "qpsolvers",
        "quadprog",
        "scikit-fuzzy",
        "scikit-learn",
        "scipy",
        "tensorflow",
        "tensorflow-probability"
    ],

    python_requires='>=3',
)
