# ConSCompF: Consistency-focused Similarity Comparison Framework

Python implementation of ConSCompF - LLM similarity comparison framework that accounts for instruction consistency proposed in the original [paper](https://doi.org/10.1613/jair.1.17028).

![LLM comprarison using ConSCompF](https://github.com/alex-karev/conscompf/blob/97c84359e2f581e2991901734f4a27af710dbeef/assets/screenshot.png)

## Features

- Generates LLM similarity matrices and compresses them using PCA.
- Can be used in few-shot scenarios.
- Supports multiple input formats including lists, HF datasets, and pandas DataFrames.
- Supports different return types including lists, PyTorch tensors, and pandas DataFrames.
- Supports embedding caching.

## Installation

Currently, there is no package available on pip. You can build and install it manually:

```bash
git clone https://github.com/alex-karev/conscompf
cd conscompf
python -m build .
pip install .
```

## Usage

```python
from conscompf import ConSCompF

conscompf = ConSCompF(quiet=True)

data: list[dict[str, list[str]]] = [
    {
        "model1": [
            "Text 1...",
            "Text 2...",
        ], 
        "model2": [
            "Text 1...",
            "Text 2...",
        ], 
    }, {
        "model1": [...],
        "model2": [...]
    }, ...
] # Or use HF dataset with a similar structure

out = conscompf(data, return_type="df") # Available return types: pt, df, list

print(out["sim_matrix"])
print(out["pca"])
print(out["consistency"])
```

The same minimalistic example, but with real data can be found in [examples/simple.py](https://github.com/alex-karev/conscompf/tree/main/examples/simple.py).

More examples are available in [examples](https://github.com/alex-karev/conscompf/tree/main/examples) directory.

For a full list of available functions and arguments use the documentation:

```bash
pydoc conscompf.ConSCompF
```

## Citation

This project is currently contributed by Alexey Karev and Dong Xu from School of Computer Engineering and Science of Shanghai University.

If you find our work valuable, please cite:

```
 @article{
    Karev_Xu_2025, 
    title={ConSCompF: Consistency-focused Similarity Comparison Framework for Generative Large Language Models}, 
    volume={82}, 
    ISSN={1076-9757}, 
    DOI={10.1613/jair.1.17028},
    journal={Journal of Artificial Intelligence Research}, 
    author={Karev, Alexey and Xu, Dong}, 
    year={2025}, 
    month=mar, 
    pages={1325–1347} 
}
```

The original dataset used during the experiments described in the original paper is available [here](https://huggingface.co/datasets/alex-karev/llm-comparison).

## Contribution

Feel free to fork this repo and make pull requests.

## Lisense

Free to use under Apacha 2.0. See LICENSE for more information.
