# FMAT <img src="man/figures/logo.png" align="right" height="160"/>

😷 The Fill-Mask Association Test (掩码填空联系测验).

The *Fill-Mask Association Test* (FMAT) is an integrative and probability-based method using [BERT Models] to measure conceptual associations (e.g., attitudes, biases, stereotypes, social norms, cultural values) as *propositions* in natural language ([Bao, 2024, *JPSP*](https://doi.org/10.1037/pspa0000396)).

![FMAT Workflow](https://psychbruce.github.io/img/FMAT-Workflow.png)

<!-- badges: start -->

[![CRAN-Version](https://www.r-pkg.org/badges/version/FMAT?color=red)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Version](https://img.shields.io/github/r-package/v/psychbruce/FMAT?label=GitHub&color=orange)](https://github.com/psychbruce/FMAT) [![R-CMD-check](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/psychbruce/FMAT/actions/workflows/R-CMD-check.yaml) [![CRAN-Downloads](https://cranlogs.r-pkg.org/badges/grand-total/FMAT)](https://CRAN.R-project.org/package=FMAT) [![GitHub-Stars](https://img.shields.io/github/stars/psychbruce/FMAT?style=social)](https://github.com/psychbruce/FMAT/stargazers)

<!-- badges: end -->

<img src="https://psychbruce.github.io/img/CC-BY-NC-SA.jpg" width="120px" height="42px"/>

## Author

Han-Wu-Shuang (Bruce) Bao 包寒吴霜

📬 [baohws\@foxmail.com](mailto:baohws@foxmail.com)

📋 [psychbruce.github.io](https://psychbruce.github.io)

## Citation

-   Bao, H.-W.-S. (2023). *FMAT: The Fill-Mask Association Test*. <https://CRAN.R-project.org/package=FMAT>
    -   *Note*: This is the original citation format. Please refer to the information when you `library(FMAT)` for the APA-7 format of your installed version.
-   Bao, H.-W.-S. (in press). The Fill-Mask Association Test (FMAT): Measuring propositions in natural language. *Journal of Personality and Social Psychology*. <https://doi.org/10.1037/pspa0000396>

## Installation

To use the FMAT, the R package `FMAT` and two Python packages (`transformers` and `torch`) all need to be installed.

### (1) R Package

``` r
## Method 1: Install from CRAN
install.packages("FMAT")

## Method 2: Install from GitHub
install.packages("devtools")
devtools::install_github("psychbruce/FMAT", force=TRUE)
```

### (2) Python Environment and Packages

#### Step 1

Install [Anaconda](https://www.anaconda.com/download) (a recommended package manager which automatically installs Python, Python IDEs like Spyder, and a large list of necessary [Python package dependencies](https://docs.anaconda.com/free/anaconda/pkg-docs/)).

#### Step 2

Specify the Python interpreter in RStudio.

> RStudio → Tools → Global/Project Options\
> → Python → Select → **Conda Environments**\
> → Choose **".../Anaconda3/python.exe"**

#### Step 3

Install the "[transformers](https://huggingface.co/docs/transformers/installation)" and "[torch](https://pytorch.org/get-started/locally/)" Python packages.\
(Windows Command / Anaconda Prompt / RStudio Terminal)

```         
pip install transformers torch
```

See [Guidance for GPU Acceleration] if you have an NVIDIA GPU device on your PC and want to use GPU to accelerate the pipeline.

#### Alternative Approach

(Not suggested) Besides the pip/conda installation in the *Conda Environment*, you might instead create and use a *Virtual Environment* (see R code below with the `reticulate` package), but then you need to specify the Python interpreter as **"\~/.virtualenvs/r-reticulate/Scripts/python.exe"** in RStudio.

``` r
## DON'T RUN THIS UNLESS YOU PREFER VIRTUAL ENVIRONMENT
library(reticulate)
# install_python()
virtualenv_create()
virtualenv_install(packages=c("transformers", "torch"))
```

## Guidance for FMAT

### FMAT Step 1: Query Design

Design queries that conceptually represent the constructs you would measure (see [Bao, 2024, *JPSP*](https://doi.org/10.1037/pspa0000396) for how to design queries).

Use `FMAT_query()` and/or `FMAT_query_bind()` to prepare a `data.table` of queries.

### FMAT Step 2: Model Loading

Use `BERT_download()` and `FMAT_load()` to (down)load [BERT models]. Model files are saved to your local folder "%USERPROFILE%/.cache/huggingface". A full list of BERT-family models are available at [Hugging Face](https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers).

### FMAT Step 3: Model Processing

Use `FMAT_run()` to get raw data (probability estimates) for further analysis.

Several steps of pre-processing have been included in the function for easier use (see `FMAT_run()` for details).

-   For BERT variants using `<mask>` rather than `[MASK]` as the mask token, the input query will be *automatically* modified so that users can always use `[MASK]` in query design.
-   For some BERT variants, special prefix characters such as `\u0120` and `\u2581` will be *automatically* added to match the whole words (rather than subwords) for `[MASK]`.

### Notes

-   Improvements are ongoing, especially for adaptation to more diverse (less popular) BERT models.
-   If you find bugs or have problems using the functions, please report them at [GitHub Issues](https://github.com/psychbruce/FMAT/issues) or send me an email.

## Guidance for GPU Acceleration

### NVIDIA GPU Acceleration

By default, the `FMAT` package uses CPU to enable the functionality for all users. But for advanced users who want to accelerate the pipeline with GPU, the `FMAT_load()` function now supports using a GPU device, which may perform **3x faster** than CPU.

### Step 1

Ensure that you have an NVIDIA GPU device (e.g., GeForce RTX Series) and an NVIDIA GPU driver installed on your system.

### Step 2

Install PyTorch (Python `torch` package) with CUDA support (<https://pytorch.org/get-started/locally/>).

-   CUDA is only available on Windows and Linux, but not on MacOS.
-   If you have installed a version of `torch` without CUDA support, please first uninstall it (command: `pip uninstall torch`) and then install the suggested one.
-   You may also install the corresponding version of CUDA Toolkit (e.g., for the `torch` version supporting CUDA 12.1, the same version of [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) may also be installed).

Example code for installing PyTorch with CUDA support:\
(Windows Command / Anaconda Prompt / RStudio Terminal)

```         
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3

Check with the `FMAT` package.

``` r
library(FMAT)
# BERT_download("bert-base-uncased")
model = FMAT_load("bert-base-uncased", gpu=TRUE)
```

```         
ℹ Device Info:

Python Environment:
Package       Version
transformers  4.38.2
torch         2.2.1+cu121

NVIDIA GPU CUDA Support:
CUDA Enabled: TRUE
CUDA Version: 12.1
GPU (Device): NVIDIA GeForce RTX 2050

Loading models from C:/Users/Bruce/.cache/huggingface/hub...
✔ bert-base-uncased (1.1s) - GPU (device id = 0)
```

(Tested 2024/03 on the developer's computer: HP Probook 450 G10 Notebook PC)

## BERT Models

The reliability and validity of the following 12 representative BERT models have been established in my research articles, but future work is needed to examine the performance of other models.

(model name on Hugging Face - downloaded model file size)

1.  [bert-base-uncased](https://huggingface.co/bert-base-uncased) (420 MB)
2.  [bert-base-cased](https://huggingface.co/bert-base-cased) (416 MB)
3.  [bert-large-uncased](https://huggingface.co/bert-large-uncased) (1283 MB)
4.  [bert-large-cased](https://huggingface.co/bert-large-cased) (1277 MB)
5.  [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) (256 MB)
6.  [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) (251 MB)
7.  [albert-base-v1](https://huggingface.co/albert-base-v1) (45 MB)
8.  [albert-base-v2](https://huggingface.co/albert-base-v2) (45 MB)
9.  [roberta-base](https://huggingface.co/roberta-base) (476 MB)
10. [distilroberta-base](https://huggingface.co/distilroberta-base) (316 MB)
11. [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base) (517 MB)
12. [vinai/bertweet-large](https://huggingface.co/vinai/bertweet-large) (1356 MB)

If you are new to [BERT](https://arxiv.org/abs/1810.04805), please read:

-   [BERT Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
-   [Breaking BERT Down](https://towardsdatascience.com/breaking-bert-down-430461f60efb)
-   [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
-   [Visual Guide to BERT](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
-   [BERT Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
-   [What is Fill-Mask?](https://huggingface.co/tasks/fill-mask)

``` r
library(FMAT)
model.names = c(
  "bert-base-uncased",
  "bert-base-cased",
  "bert-large-uncased",
  "bert-large-cased",
  "distilbert-base-uncased",
  "distilbert-base-cased",
  "albert-base-v1",
  "albert-base-v2",
  "roberta-base",
  "distilroberta-base",
  "vinai/bertweet-base",
  "vinai/bertweet-large"
)
BERT_download(model.names)
```

```         
ℹ Device Info:

Python Environment:
Package       Version
transformers  4.38.2
torch         2.2.1+cu121

NVIDIA GPU CUDA Support:
CUDA Enabled: TRUE
CUDA Version: 12.1
GPU (Device): NVIDIA GeForce RTX 2050


── Downloading model "bert-base-uncased" ───────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 570/570 [00:00<00:00, 113kB/s]
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 48.0/48.0 [00:00<?, ?B/s]
vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 1.37MB/s]
tokenizer.json: 100%|██████████| 466k/466k [00:00<00:00, 3.94MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 440M/440M [01:21<00:00, 5.40MB/s] 
✔ Successfully downloaded model "bert-base-uncased"

── Downloading model "bert-base-cased" ─────────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 570/570 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 49.0/49.0 [00:00<00:00, 8.18kB/s]
vocab.txt: 100%|██████████| 213k/213k [00:00<00:00, 1.30MB/s]
tokenizer.json: 100%|██████████| 436k/436k [00:00<00:00, 3.67MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 436M/436M [01:20<00:00, 5.41MB/s] 
✔ Successfully downloaded model "bert-base-cased"

── Downloading model "bert-large-uncased" ──────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 571/571 [00:00<00:00, 143kB/s]
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 48.0/48.0 [00:00<00:00, 12.0kB/s]
vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 6.04MB/s]
tokenizer.json: 100%|██████████| 466k/466k [00:00<00:00, 1.57MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 1.34G/1.34G [04:09<00:00, 5.39MB/s]
✔ Successfully downloaded model "bert-large-uncased"

── Downloading model "bert-large-cased" ────────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 762/762 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 49.0/49.0 [00:00<?, ?B/s]
vocab.txt: 100%|██████████| 213k/213k [00:00<00:00, 2.14MB/s]
tokenizer.json: 100%|██████████| 436k/436k [00:00<00:00, 1.75MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 1.34G/1.34G [04:08<00:00, 5.38MB/s]
✔ Successfully downloaded model "bert-large-cased"

── Downloading model "distilbert-base-uncased" ─────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 483/483 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 28.0/28.0 [00:00<?, ?B/s]
vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 1.36MB/s]
tokenizer.json: 100%|██████████| 466k/466k [00:00<00:00, 1.82MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 268M/268M [00:51<00:00, 5.24MB/s] 
✔ Successfully downloaded model "distilbert-base-uncased"

── Downloading model "distilbert-base-cased" ───────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 465/465 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 29.0/29.0 [00:00<?, ?B/s]
vocab.txt: 100%|██████████| 213k/213k [00:00<00:00, 1.34MB/s]
tokenizer.json: 100%|██████████| 436k/436k [00:00<00:00, 4.20MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 263M/263M [00:49<00:00, 5.36MB/s] 
✔ Successfully downloaded model "distilbert-base-cased"

── Downloading model "albert-base-v1" ──────────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 684/684 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 25.0/25.0 [00:00<00:00, 1.65kB/s]
spiece.model: 100%|██████████| 760k/760k [00:00<00:00, 4.58MB/s]
tokenizer.json: 100%|██████████| 1.31M/1.31M [00:00<00:00, 3.09MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 47.4M/47.4M [00:09<00:00, 5.07MB/s]
✔ Successfully downloaded model "albert-base-v1"

── Downloading model "albert-base-v2" ──────────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 684/684 [00:00<00:00, 45.5kB/s]
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 25.0/25.0 [00:00<?, ?B/s]
spiece.model: 100%|██████████| 760k/760k [00:00<00:00, 2.13MB/s]
tokenizer.json: 100%|██████████| 1.31M/1.31M [00:00<00:00, 5.66MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 47.4M/47.4M [00:08<00:00, 5.51MB/s]
✔ Successfully downloaded model "albert-base-v2"

── Downloading model "roberta-base" ────────────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 481/481 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 25.0/25.0 [00:00<?, ?B/s]
vocab.json: 100%|██████████| 899k/899k [00:00<00:00, 5.73MB/s]
merges.txt: 100%|██████████| 456k/456k [00:00<00:00, 6.16MB/s]
tokenizer.json: 100%|██████████| 1.36M/1.36M [00:00<00:00, 5.50MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 499M/499M [01:32<00:00, 5.38MB/s] 
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
✔ Successfully downloaded model "roberta-base"

── Downloading model "distilroberta-base" ──────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 480/480 [00:00<00:00, 30.7kB/s]
→ (2) Downloading tokenizer...
tokenizer_config.json: 100%|██████████| 25.0/25.0 [00:00<00:00, 7.98kB/s]
vocab.json: 100%|██████████| 899k/899k [00:00<00:00, 5.18MB/s]
merges.txt: 100%|██████████| 456k/456k [00:00<00:00, 5.71MB/s]
tokenizer.json: 100%|██████████| 1.36M/1.36M [00:00<00:00, 3.83MB/s]
→ (3) Downloading model...
model.safetensors: 100%|██████████| 331M/331M [01:01<00:00, 5.39MB/s] 
✔ Successfully downloaded model "distilroberta-base"

── Downloading model "vinai/bertweet-base" ─────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 558/558 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
vocab.txt: 100%|██████████| 843k/843k [00:00<00:00, 5.56MB/s]
bpe.codes: 100%|██████████| 1.08M/1.08M [00:00<00:00, 5.55MB/s]
tokenizer.json: 100%|██████████| 2.91M/2.91M [00:00<00:00, 5.50MB/s]
emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3 install emoji==0.6.0
→ (3) Downloading model...
pytorch_model.bin: 100%|██████████| 543M/543M [01:40<00:00, 5.39MB/s] 
✔ Successfully downloaded model "vinai/bertweet-base"

── Downloading model "vinai/bertweet-large" ────────────────────────────────────────
→ (1) Downloading configuration...
config.json: 100%|██████████| 614/614 [00:00<?, ?B/s] 
→ (2) Downloading tokenizer...
vocab.json: 100%|██████████| 899k/899k [00:00<00:00, 5.59MB/s]
merges.txt: 100%|██████████| 456k/456k [00:00<00:00, 5.04MB/s]
tokenizer.json: 100%|██████████| 1.36M/1.36M [00:00<00:00, 5.42MB/s]
→ (3) Downloading model...
pytorch_model.bin: 100%|██████████| 1.42G/1.42G [04:23<00:00, 5.40MB/s]
Some weights of RobertaModel were not initialized from the model checkpoint at vinai/bertweet-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
✔ Successfully downloaded model "vinai/bertweet-large"

── Downloaded models: ──

                           Size
albert-base-v1            45 MB
albert-base-v2            45 MB
bert-base-cased          416 MB
bert-base-uncased        420 MB
bert-large-cased        1277 MB
bert-large-uncased      1283 MB
distilbert-base-cased    251 MB
distilbert-base-uncased  256 MB
distilroberta-base       316 MB
roberta-base             476 MB
vinai/bertweet-base      517 MB
vinai/bertweet-large    1356 MB

✔ Downloaded models saved at C:/Users/Bruce/.cache/huggingface/hub (6.52 GB)
```

(Tested 2024/03 on the developer's computer: HP Probook 450 G10 Notebook PC)

## Related Packages

While the FMAT is an innovative method for the *computational intelligent* analysis of psychology and society, you may also seek for an integrative toolbox for other text-analytic methods. Another R package I developed---[PsychWordVec](https://psychbruce.github.io/PsychWordVec/)---is useful and user-friendly for word embedding analysis (e.g., the Word Embedding Association Test, WEAT). Please refer to its documentation and feel free to use it.
