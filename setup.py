from setuptools import setup


with open('README.rst') as f:
    long_description = f.read()


setup(
    name='rpcdataloader',
    version='0.0.1',
    author='Nicolas Granger',
    author_email='nicolas.granger@cea.fr',
    description='A Dataloader using rpc-based workers',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license_files=['LICENSE.txt'],
    install_requires=[
        'numpy',
        'torch',
        'tblib',
    ],
    tests_require=[
        'pytest'
    ],
    extras_require={
        'test': ['pytest'],
        'doc': ['sphinx', 'sphinx-autodoc-typehints', 'sphinx-rtd-theme']
    },
    classifiers=[
        'License :: CeCILL-C Free Software License Agreement (CECILL-C)',
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)