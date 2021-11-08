# A Temporal Variational Model for Story Generation

This is the code repository for the following [[paper]](https://arxiv.org/abs/2109.06807):

## Abstract

Recent language models can generate interesting and grammatically correct text in story generation but often lack plot development and long-term coherence. This paper experiments with a latent vector planning approach based on a TD-VAE (Temporal Difference Variational Autoencoder), using the model for conditioning and reranking for text generation. The results demonstrate strong performance in automatic cloze and swapping evaluations. The human judgments show stories generated with TD-VAE reranking improve on a GPT-2 medium baseline and show comparable performance to a hierarchical LSTM reranking model. Conditioning on the latent vectors proves disappointing and deteriorates performance in human evaluation because it reduces the diversity of generation, and the models don't learn to progress the narrative. This highlights an important difference between technical task performance (e.g. cloze) and generating interesting stories.

## Citation

```
@article{DBLP:journals/corr/abs-2109-06807,
  author    = {David Wilmot and
               Frank Keller},
  title     = {A Temporal Variational Model for Story Generation},
  journal   = {CoRR},
  volume    = {abs/2109.06807},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.06807},
  eprinttype = {arXiv},
  eprint    = {2109.06807},
  timestamp = {Tue, 21 Sep 2021 17:46:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2109-06807.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
