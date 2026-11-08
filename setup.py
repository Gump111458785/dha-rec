from setuptools import setup, find_packages

setup(
    name="dha_rec",
    version="0.1.0",
    description="Dual-Horizon Attentive Recommendation Network for Sequential Item Prediction",
    author="Anonymous Authors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
    ],
)
