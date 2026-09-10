from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="BrainSAM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "numpy",
        "tqdm",
        "hydra-core>=1.3.2",
        "omegaconf",
        "iopath>=0.1.10",
        "pillow",
        "matplotlib",
        "opencv-python",
        "scipy",
        "pandas",
        "openpyxl",
        "nibabel",
        "SimpleITK",
    ],
    extras_require={
        "web": ["fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2", "websockets"],
        "gui": ["PyQt5"],
        "train": ["submitit", "tensordict", "wandb", "huggingface_hub"],
    },
    description="BrainSAM: a SAM2-based model for brain structure segmentation across species/modalities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.10",
)