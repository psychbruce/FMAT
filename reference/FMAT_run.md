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
  add.method = c("sum", "mean"),
  add.verbose = TRUE,
  pattern.special = list(uncased = "uncased|albert|electra|muhtasham", prefix.u2581 =
    "albert|xlm-roberta|xlnet", prefix.u2581.excl = "chinese", prefix.u0120 =
    "roberta|bart|deberta|bertweet-large", prefix.u0120.excl = "chinese|xlm-|kornosk/"),
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
  vocabulary? It only temporarily adds tokens for tasks but does not
  change the raw model file. Defaults to `FALSE`.

- add.method:

  Method used to produce the token embeddings of newly added tokens. Can
  be `"sum"` (default) or `"mean"` of subword token embeddings.

- add.verbose:

  Print composition information of new tokens (for out-of-vocabulary
  words or phrases)? Defaults to `TRUE`.

- pattern.special:

  Regular expression patterns (matching model names) for special model
  cases that are uncased or require a special prefix character in
  certain situations.

  **WARNING**: As the developer is not able to check all models, users
  are responsible for checking the models they would use and for
  modifying this argument if necessary.

  - `prefix.u2581`: adding prefix `\u2581` for all mask words

  - `prefix.u0120`: adding prefix `\u0120` for only non-starting mask
    words

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

  - It is *NOT SUGGESTED* to directly interpret the raw probabilities
    because the *contrast* between a pair of probabilities is more
    interpretable. See
    [`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md).

## Details

The function automatically adjusts for the compatibility of tokens used
in certain models: (1) for uncased models (e.g., ALBERT), it turns
tokens to lowercase; (2) for models that use `<mask>` rather than
`[MASK]`, it automatically uses the corrected mask token; (3) for models
that require a prefix to estimate whole words than subwords (e.g.,
ALBERT, RoBERTa), it adds a certain prefix (usually a white space;
\u2581 for ALBERT and XLM-RoBERTa, \u0120 for RoBERTa and
DistilRoBERTa).

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
