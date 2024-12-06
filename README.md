## Let's Ask GNN: Empowering Large Language Models for Graph In-Context Learning


This repository contains the implementation of the paper "Let's Ask GNN: Empowering Large Language Models for Graph In-Context Learning". The goal of this work is to combine Graph Neural Networks (GNNs) with Large Language Models (LLMs) to enhance graph understanding and relational reasoning capabilities in a variety of graph-based tasks.




### scripts

To train the GNN using the SAGE model on the OGBN-Arxiv dataset, use the following command:

```python
python gen_result_local_llm.py  --stru sage   --dataset ogbn-arxiv --ratio 0.05

```

- `--stru`: The type of GNN architecture to use (e.g., `sage` for GraphSAGE).
- `--dataset`: The graph dataset to use (e.g., `ogbn-arxiv`).
- `--ratio`: The proportion of data to use (e.g., 0.05 represents 5%).



After training the GNN and generating embeddings, the next step is to perform inference with a Large Language Model (LLM). Use the following command for inference:


```python
python llm_inference.py  --stru sage   --dataset ogbn-arxiv --ratio 0.05  --llm_model qwen

```

- `--stru`: Specifies the GNN model (e.g., `sage`).
- `--dataset`: The graph dataset to use (e.g., `ogbn-arxiv`).
- `--ratio`: The proportion of the dataset to use (same as in the training step).
- `--llm_model`: The LLM model to use (e.g., `qwen`).
