#!/usr/bin/env python

# THIS CODE IS ADAPTED FROM (AN OLD VERSION OF) https://github.com/dfm/extending-jax

import codecs
import os
import subprocess
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))

srcpath=f"{HERE}/src"
if os.path.exists(srcpath):
    shutil.rmtree(srcpath)
os.makedirs(srcpath)

def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


# This custom class for building the extensions uses CMake to compile. You
# don't have to use CMake for this task, but I found it to be the easiest when
# compiling ops with GPU support since setuptools doesn't have great CUDA
# support.
class CMakeBuildExt(build_ext):
    def build_extensions(self):
        # First: configure CMake build
        import platform
        import sys
        import distutils.sysconfig

        import pybind11

        # Work out the relevant Python paths to pass to CMake, adapted from the
        # PyTorch build system
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                distutils.sysconfig.get_config_var("prefix"),
                distutils.sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    distutils.sysconfig.get_config_var("VERSION"),
                )
        else:
            cmake_python_library = "{}/{}".format(
                distutils.sysconfig.get_config_var("LIBDIR"),
                distutils.sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = distutils.sysconfig.get_python_inc()

        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
        ]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )


extensions = [
    Extension(
        "jaxpops.cpu", # Name
        ["lib/cpu.cc"],
    ),
]


setup(
    name="jaxpops",
    version="0.0.2",
    author="Mauro Rigo",
    author_email="mrigo@sissa.it",
    description=(
        "Parallel CIC painter and readout and parallel FFT and inverse"
        "using FFTW, written to be used in JAX."
    ),
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "jax>=0.4.16,<=0.4.25",
        "jaxlib>=0.4.16,<=0.4.25",
        "numpy<2",
        "mpi4jax>=0.4.0"
    ],
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
