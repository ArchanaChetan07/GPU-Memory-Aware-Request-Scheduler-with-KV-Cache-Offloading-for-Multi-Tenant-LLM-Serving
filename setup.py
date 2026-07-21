"""Setup script for Request-Level Context Swapping.

CPU-only by default. To build the CUDA extension (requires nvcc + PyTorch):

    CONTEXT_SWAP_FORCE_CUDA=1 pip install -e .
"""

import os

from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

if os.environ.get("CONTEXT_SWAP_FORCE_CUDA"):
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="context_swapping._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/context_swap_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="context-swapping",
    version="0.1.0",
    description="Request-level context swapping for multi-tenant LLM serving",
    author="Context Swapping Team",
    license="Apache-2.0",
    python_requires=">=3.10",
    # src/ IS the package: find_packages(where="src") would look for
    # subpackages inside it and find nothing, installing zero code
    packages=["context_swapping"],
    package_dir={"context_swapping": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
        ],
        "cuda": ["torch>=2.0.0"],
        "vllm": [
            "vllm>=0.3.0",
            "transformers>=4.30.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
