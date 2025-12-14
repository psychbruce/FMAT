# Get basic information of BERT models.

Get basic information of BERT models.

## Usage

``` r
BERT_info(models = NULL)
```

## Arguments

- models:

  A character vector of model names at
  [HuggingFace](https://huggingface.co/models).

## Value

A data.table:

- model name

- model type

- number of parameters

- vocabulary size (of input token embeddings)

- embedding dimensions (of input token embeddings)

- hidden layers

- attention heads

- \[MASK\] token

## See also

[`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)

[`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)

## Examples

``` r
if (FALSE) { # \dontrun{
models = c("bert-base-uncased", "bert-base-cased")
BERT_info(models)

BERT_info()  # information of all downloaded models
# speed: ~1.2s/model for first use; <1s afterwards
} # }
```
