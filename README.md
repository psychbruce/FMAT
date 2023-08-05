# FMAT <img src="man/figures/logo.png" align="right" height="160"/>

😷 The Fill-Mask Association Test (掩码填空联系测验).

The *Fill-Mask Association Test* (FMAT) is an integrative, versatile, and probability-based method that uses Masked Language Models ([BERT](https://arxiv.org/abs/1810.04805)) to measure conceptual associations (e.g., attitudes, biases, stereotypes) as propositional representations in natural language.

A full list of BERT-family models are available at [Hugging Face](https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers). Use the `FMAT_load()` function to download and load specific BERT models. All downloaded model files are saved at your local folder "C:/Users/[YourUserName]/.cache/".

Several necessary pre-processing steps have been designed in the functions for easier and more direct use (see `FMAT_run()` for details).

-   For those BERT variants using `<mask>` rather than `[MASK]` as the mask token, the input query will be *automatically* modified so that users can always use `[MASK]` in query design.
-   For some BERT variants, special prefix characters such as `\u0120` and `\u2581` will be *automatically* added to match the whole words (rather than subwords) for `[MASK]`.

Improvements are still needed. If you find bugs or have problems using the functions, please report them at [GitHub Issues](https://github.com/psychbruce/FMAT/issues) or send me an email.

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/FMAT?color=red)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/FMAT?label=GitHub&color=orange)](https://github.com/psychbruce/FMAT) [![R-CMD-check](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/FMAT)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/FMAT?style=social)](https://github.com/psychbruce/FMAT/stargazers)

<!-- badges: end -->

<img src="https://s1.ax1x.com/2020/07/28/aAjUJg.jpg" width="120px" height="42px"/>

## Author

Han-Wu-Shuang (Bruce) Bao 包寒吴霜

Email: [baohws\@foxmail.com](mailto:baohws@foxmail.com)

Homepage: [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H.-W.-S. (2023). *The Fill-Mask Association Test (FMAT)*. R package version 0.1.x. <https://CRAN.R-project.org/package=FMAT>
-   Bao, H.-W.-S. (2023). *The Fill-Mask Association Test (FMAT): Using AI language models to better understand society and culture* [Manuscript submitted for publication].

## Installation

```{r}
## Method 1: Install from CRAN
install.packages("FMAT")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/FMAT", force=TRUE)
```

Since this package uses the "[reticulate](https://CRAN.R-project.org/package=reticulate)" package for an R interface to the "transformers" Python module, you need also to install both [Python](https://www.anaconda.com/) (with Anaconda) and the "[transformers](https://huggingface.co/docs/transformers/installation)" module (with command `pip install transformers`) in your computer.

## BERT Models

The reliability and validity of the following 12 representative BERT models have been established in my research articles, but future work is needed to examine the performance of other models.

(model name on Hugging Face - downloaded file size)

1.  [bert-base-uncased](https://huggingface.co/bert-base-uncased) (420MB)
2.  [bert-base-cased](https://huggingface.co/bert-base-cased) (416MB)
3.  [bert-large-uncased](https://huggingface.co/bert-large-uncased) (1.25GB)
4.  [bert-large-cased](https://huggingface.co/bert-large-cased) (1.25GB)
5.  [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) (256MB)
6.  [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) (251MB)
7.  [albert-base-v1](https://huggingface.co/albert-base-v1) (45.2MB)
8.  [albert-base-v2](https://huggingface.co/albert-base-v2) (45.2MB)
9.  [roberta-base](https://huggingface.co/roberta-base) (478MB)
10. [distilroberta-base](https://huggingface.co/distilroberta-base) (316MB)
11. [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base) (517MB)
12. [vinai/bertweet-large](https://huggingface.co/vinai/bertweet-large) (1.32GB)

If you are new to [BERT](https://arxiv.org/abs/1810.04805), please read:

-   [BERT Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
-   [Breaking BERT Down](https://towardsdatascience.com/breaking-bert-down-430461f60efb)
-   [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
-   [Visual Guide to BERT](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
-   [BERT Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
-   [What is Fill-Mask?](https://huggingface.co/tasks/fill-mask)

## Related Packages

While the FMAT is an innovative method for *computational intelligent* analysis of psychology and society, you may also seek for an integrative toolbox for other text-analytic methods. Another R package I developed---[PsychWordVec](https://psychbruce.github.io/PsychWordVec/)---is one of the most useful and user-friendly package for word embedding analysis (e.g., the Word Embedding Association Test, WEAT). Please refer to its documentation and feel free to use it.
