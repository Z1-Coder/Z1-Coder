# Trajectory Data

## Downloading Dataset 
Following are the instructions to download our trajectory dataset.

## HuggingFace
The datasets are available on [HuggingFace](https://huggingface.co/Z1-Coder). Download it as follows:
```python
from datasets import load_dataset
ds_stage1 = load_dataset("Z1-Coder/Z1Coder-Evol-CoT-110K")
ds_stage2 = load_dataset("Z1-Coder/Z1Coder-SelfInvoking-CoT-20K")
```