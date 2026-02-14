from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ternary-sd",
    version="0.1.0",
    description="Ternary quantization for Stable Diffusion models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)