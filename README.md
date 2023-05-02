# FMAT <img src="man/figures/logo.png" align="right" height="160"/>

😷 The Fill-Mask Association Test.

The *Fill-Mask Association Test* (FMAT) is a versatile, probability-based method that employs Masked Language Models ([BERT](https://arxiv.org/abs/1810.04805)) to measure conceptual associations (e.g., attitudes, biases, stereotypes) in natural language.

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/FMAT?color=red)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/FMAT?label=GitHub&color=orange)](https://github.com/psychbruce/FMAT) [![R-CMD-check](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/FMAT)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/FMAT?style=social)](https://github.com/psychbruce/FMAT/stargazers)

<!-- badges: end -->

<img src="https://s1.ax1x.com/2020/07/28/aAjUJg.jpg" width="120px" height="42px"/>

## Author

Han-Wu-Shuang (Bruce) Bao 包寒吴霜

Email: [baohws\@foxmail.com](mailto:baohws@foxmail.com)

Homepage: [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H.-W.-S. (2023). *Unmasking human society in natural language: The Fill-Mask Association Test (FMAT)*. Manuscript in preparation.

## Installation

```{r}
## Method 1: Install from CRAN
install.packages("FMAT")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/FMAT", force=TRUE)
```

## BERT Models

If you are new to [BERT](https://arxiv.org/abs/1810.04805), please read:

-   [BERT Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
-   [Breaking BERT Down](https://towardsdatascience.com/breaking-bert-down-430461f60efb)
-   [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
-   [Visual Guide to BERT](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
-   [BERT Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
-   [What is Fill-Mask?](https://huggingface.co/tasks/fill-mask)
