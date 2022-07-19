from setuptools import setup, find_packages  # , Extension

# from distutils.spawn import find_executable
# import os


with open("README.rst", "r") as f:
    long_description = f.read()


# detect cuda installation
# if 'CUDA_HOME' in os.environ:
#     CUDA_HOME = os.environ['CUDA_HOME']
# elif find_executable('nvcc') is not None:
#     CUDA_HOME = os.path.dirname(find_executable('nvcc'))
# elif os.path.isdir('/usr/loca/cuda'):
#     CUDA_HOME = '/usr/loca/cuda'
# else:
#    print("CUDA_HOME environment is not set or does not exist".format(CUDA_HOME))
#    exit(-1)


setup(
    name="rpcdataloader",
    version="0.0.1",
    author="Nicolas Granger",
    author_email="nicolas.granger@cea.fr",
    description="A Dataloader using rpc-based workers",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license_files=["LICENSE.txt"],
    install_requires=["tblib", 'typing;python_version<"3.9"'],
    tests_require=["pytest"],
    extras_require={
        "pytorch": ["torch", "numpy"],
        "test": ["pytest"],
        "doc": ["sphinx", "sphinx-rtd-theme"],
    },
    # ext_modules=[
    #     Extension(
    #         name='rpcdataloader.pinned_buffer',
    #         sources=['rpcdataloader/pinned_buffer.c'],
    #         include_dirs=[os.path.join(CUDA_HOME, 'include')],
    #         extra_link_args=['-L' + os.path.join(CUDA_HOME, 'lib64'), '-lcudart'])
    # ],
    packages=find_packages(where="rpcdataloader"),
    classifiers=[
        "License :: CeCILL-C Free Software License Agreement (CECILL-C)",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
