from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="soft_moe",
    packages=find_packages(),
    version="0.0.1",
    license="Apache-2.0",
    description="PyTorch implementation of 'From Sparse to Soft Mixtures of Experts'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ben Conrad",
    author_email="benwconrad@proton.me",
    url="https://github.com/bwconrad/soft-moe",
    keywords=[
        "transformers",
        "artificial intelligence",
        "computer vision",
        "deep learning",
    ],
    install_requires=[
        "timm >= 0.9.2",
        "torch >= 2.0.1",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
)
