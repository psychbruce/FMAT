# Check if mask words are in the model vocabulary.

Check if mask words are in the model vocabulary.

## Usage

``` r
BERT_vocab(
  models,
  mask.words,
  add.tokens = FALSE,
  add.method = c("mean", "sum"),
  add.verbose = TRUE
)
```

## Arguments

- models:

  A character vector of model names at
  [HuggingFace](https://huggingface.co/models).

- mask.words:

  Option words filling in the mask.

- add.tokens:

  Add new tokens (for out-of-vocabulary words or phrases) to model
  vocabulary? It only temporarily adds tokens for tasks but does not
  change the raw model file. Defaults to `FALSE`.

- add.method:

  Method used to produce the token embeddings of appended tokens. Can be
  `"mean"` (default) or `"sum"` of subword token embeddings.

- add.verbose:

  Print composition information of new tokens (for out-of-vocabulary
  words or phrases)? Defaults to `TRUE`.

## Value

A data.table of model name, mask word, real token (replaced if out of
vocabulary), and token id (0~N).

## See also

[`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)

[`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md)

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
if (FALSE) { # \dontrun{
models = c("bert-base-uncased", "bert-base-cased")
BERT_info(models)

BERT_vocab(models, c("bruce", "Bruce"))

BERT_vocab(models, 2020:2025)  # some are out-of-vocabulary
BERT_vocab(models, 2020:2025, add.tokens=TRUE)  # add vocab

BERT_vocab(models,
           c("individualism", "artificial intelligence"),
           add.tokens=TRUE)
} # }
```
