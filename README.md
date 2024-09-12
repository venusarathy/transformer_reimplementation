# Transformer Reimplementation ("Attention is All You Need")

This repository is a PyTorch reimplementation of the "Attention is All You Need" paper, including a transformer model for NLP tasks. This example trains the model on the IMDb dataset using TensorFlow Datasets for preprocessing.

## Structure
- `models/`: Contains the transformer model and embedding code.
- `training/`: Contains the training scripts.
- `data/`: Code to load and preprocess datasets.
- `notebooks/`: Jupyter notebooks for experimentation.
- `checkpoints/`: Directory to save model checkpoints.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Citations

If you use this project in your research, please cite it as follows:

### For the "Attention Is All You Need" Paper:
**APA Format:**
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, *30*. https://doi.org/10.48550/arXiv.1706.03762


```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  volume={30},
  year={2017},
  url={https://doi.org/10.48550/arXiv.1706.03762}
}
@misc{pytorch,
  title={PyTorch: An imperative style, high-performance deep learning library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Alexander and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  year={2019},
  howpublished={\url{https://github.com/pytorch/pytorch}},
}
@misc{tensorflow2015-whitepaper,
  title={TensorFlow: Large-scale machine learning on heterogeneous systems},
  author={Abadi, Mart{\'\i}n and Agarwal, Ashish and Barham, Paul and Brevdo, Eugene and Chen, Zhifeng and Citro, Craig and Corrado, Greg S and Davis, Andy and Dean, Jeffrey and Devin, Matthias and others},
  year={2015},
  howpublished={\url{https://github.com/tensorflow/tensorflow}},
}

