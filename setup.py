from setuptools import setup, find_packages

setup(
    name="multimodal_bert",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "cancer_data"
    ],
    python_requires=">=3.7",
)