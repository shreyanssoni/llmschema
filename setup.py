from setuptools import setup, find_packages

setup(
    name="llmschema",
    version="0.2.2",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "ollama>=0.1.0"
    ],
    author="Shreyans Soni",
    author_email="sonishreyans01@gmail.com",
    description="A structured response wrapper for LLMs using Pydantic.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shreyanssoni/llmschema",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
