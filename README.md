# LLM-based Keyword Extraction Does not Apply Well to Cyber-Security

## Installation

To install the dependencies when cuda is available
    python3 -m pip install -r requirements_cuda.txt --extra-index-url https://pypi.nvidia.com

Otherwise
    python3 -m pip install -r requirements.txt

## Usage

To extract the keywords

    python3 pipeline_eeke.py

To draw the manifolds with the different embeddings

    python3 manifold.py

To plot the cross correlation of the models

    python3 cross_correlation_matrix_model.py

## Remarks

- When using the scripts on large data, you can have some issue with the scripts. For example: Take too long, use too much memory, ...
- There are constant on top of the files to edit some parameters of the scripts.

## Requirements

- Python3.10
