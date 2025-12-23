# Run the fill-mask pipeline on multiple models (CPU / GPU).

Run the fill-mask pipeline on multiple models with CPU or GPU (faster
but requires an NVIDIA GPU device).

## Usage

``` r
FMAT_run(
  models,
  data,
  gpu,
  add.tokens = FALSE,
  add.verbose = FALSE,
  weight.decay = 1,
  pattern.special = special_case(),
  file = NULL,
  progress = TRUE,
  warning = TRUE,
  na.out = TRUE
)
```

## Arguments

- models:

  A character vector of model names at
  [HuggingFace](https://huggingface.co/models).

- data:

  A data.table returned from
  [`FMAT_query()`](https://psychbruce.github.io/FMAT/reference/FMAT_query.md)
  or
  [`FMAT_query_bind()`](https://psychbruce.github.io/FMAT/reference/FMAT_query_bind.md).

- gpu:

  Use GPU (3x faster than CPU) to run the fill-mask pipeline? Defaults
  to missing value that will *automatically* use available GPU (if not
  available, then use CPU). An NVIDIA GPU device (e.g., GeForce RTX
  Series) is required to use GPU. See [Guidance for GPU
  Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration).

  Options passing on to the `device` parameter in Python:

  - `FALSE`: CPU (`device = -1`).

  - `TRUE`: GPU (`device = 0`).

  - Others: passing on to
    [`transformers.pipeline(device=...)`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.device)
    which defines the device (e.g., `"cpu"`, `"cuda:0"`, or a GPU device
    id like `1`) on which the pipeline will be allocated.

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

- pattern.special:

  See
  [`special_case()`](https://psychbruce.github.io/FMAT/reference/special_case.md)
  for details.

- file:

  File name of `.RData` to save the returned data.

- progress:

  Show a progress bar? Defaults to `TRUE`.

- warning:

  Alert warning of out-of-vocabulary word(s)? Defaults to `TRUE`.

- na.out:

  Replace probabilities of out-of-vocabulary word(s) with `NA`? Defaults
  to `TRUE`.

## Value

A data.table (class `fmat`) appending `data` with these new variables:

- `model`: model name.

- `output`: complete sentence output with unmasked token.

- `token`: actual token to be filled in the blank mask (a note
  "out-of-vocabulary" will be added if the original word is not found in
  the model vocabulary).

- `prob`: (raw) conditional probability of the unmasked token given the
  provided context, estimated by the masked language model.

  - Raw probabilities should *NOT* be directly used or interpreted.
    Please use
    [`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md)
    to *contrast* between a pair of probabilities.

## Details

The function automatically adjusts for the compatibility of tokens used
in certain models: (1) for uncased models (e.g., ALBERT), it turns
tokens to lowercase; (2) for models that use `<mask>` rather than
`[MASK]`, it automatically uses the corrected mask token; (3) for models
that require a prefix to estimate whole words than subwords (e.g.,
ALBERT, RoBERTa), it adds a white space before each mask option word.
See
[`special_case()`](https://psychbruce.github.io/FMAT/reference/special_case.md)
for details.

These changes only affect the `token` variable in the returned data, but
will not affect the `M_word` variable. Thus, users may analyze data
based on the unchanged `M_word` rather than the `token`.

Note also that there may be extremely trivial differences (after 5~6
significant digits) in the raw probability estimates between using CPU
and GPU, but these differences would have little impact on main results.

## See also

[`set_cache_folder()`](https://psychbruce.github.io/FMAT/reference/set_cache_folder.md)

[`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)

[`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)

[`FMAT_query()`](https://psychbruce.github.io/FMAT/reference/FMAT_query.md)

[`FMAT_query_bind()`](https://psychbruce.github.io/FMAT/reference/FMAT_query_bind.md)

[`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md)

[`special_case()`](https://psychbruce.github.io/FMAT/reference/special_case.md)

[`weight_decay()`](https://psychbruce.github.io/FMAT/reference/weight_decay.md)

## Examples

``` r
## Running the examples requires the models downloaded

if (FALSE) { # \dontrun{
models = c("bert-base-uncased", "bert-base-cased")

query1 = FMAT_query(
  c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
  MASK = .(Male="He", Female="She"),
  TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
)
data1 = FMAT_run(models, query1)
summary(data1, target.pair=FALSE)

query2 = FMAT_query(
  "The [MASK] {ATTRIB}.",
  MASK = .(Male=c("man", "boy"),
           Female=c("woman", "girl")),
  ATTRIB = .(Masc=c("is masculine", "has a masculine personality"),
             Femi=c("is feminine", "has a feminine personality"))
)
data2 = FMAT_run(models, query2)
summary(data2, mask.pair=FALSE)
summary(data2)
} # }
```
