# Check if mask words are in the model vocabulary.

Check if mask words are in the model vocabulary.

## Usage

``` r
BERT_vocab(
  models,
  mask.words,
  add.tokens = FALSE,
  add.verbose = FALSE,
  weight.decay = 1
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
  vocabulary? Defaults to `FALSE`.

  - Default method of producing the new token embeddings is computing
    the (equally weighted) average subword token embeddings. To change
    the weights of different subwords, specify `weight.decay`.

  - It just adds tokens temporarily without changing the raw model file.

- add.verbose:

  Print subwords of each new token? Defaults to `FALSE`.

- weight.decay:

  Decay factor of relative importance of multiple subwords. Defaults to
  `1` (see
  [`weight_decay()`](https://psychbruce.github.io/FMAT/reference/weight_decay.md)
  for computational details). A smaller decay value would give greater
  weight to the former subwords than to the latter subwords. The i-th
  subword has raw weight = decay ^ i.

  - decay = 1: all subwords are **equally** important (default)

  - 0 \< decay \< 1: **first** subwords are more important

  - decay \> 1: **last** subwords are more important

  For example, decay = 0.5 would give 0.5 and 0.25 (with normalized
  weights 0.667 and 0.333) to two subwords (e.g., "individualism" =
  0.667 "individual" + 0.333 "##ism").

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
