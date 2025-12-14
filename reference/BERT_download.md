# Download and save BERT models to local cache folder.

Download and save BERT models to local cache folder
"%USERPROFILE%/.cache/huggingface".

## Usage

``` r
BERT_download(models = NULL, verbose = FALSE)
```

## Arguments

- models:

  A character vector of model names at
  [HuggingFace](https://huggingface.co/models).

- verbose:

  Alert if a model has been downloaded. Defaults to `FALSE`.

## Value

Invisibly return a data.table of basic file information of local models.

## See also

[`set_cache_folder()`](https://psychbruce.github.io/FMAT/reference/set_cache_folder.md)

[`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md)

[`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)

## Examples

``` r
if (FALSE) { # \dontrun{
models = c("bert-base-uncased", "bert-base-cased")
BERT_download(models)

BERT_download()  # check downloaded models

BERT_info()  # information of all downloaded models
} # }
```
