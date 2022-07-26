from setuptools import setup, find_packages


with open("README.rst", "r") as f:
    long_description = f.read()


setup(
    name="rpcdataloader",
    version="0.0.1",
    author="Nicolas Granger",
    author_email="nicolas.granger@cea.fr",
    description="A Dataloader using rpc-based workers",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license_files=["LICENSE.txt"],
    install_requires=[
        "tblib",
        'typing;python_version<"3.9"',
        'pickle5;python_version<"3.8"',
    ],
    tests_require=["pytest"],
    extras_require={
        "pytorch": ["torch", "numpy"],
        "test": ["pytest"],
        "doc": ["sphinx", "sphinx-rtd-theme"],
    },
    packages=find_packages(),
    classifiers=[
        "License :: CeCILL-C Free Software License Agreement (CECILL-C)",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
