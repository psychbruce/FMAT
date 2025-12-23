# Run the fill-mask pipeline and check the raw results.

This function is only for technical check. Please use
[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
for general purposes.

## Usage

``` r
fill_mask(query, model, targets = NULL, topn = 5, gpu)

fill_mask_check(query, models, targets = NULL, topn = 5, gpu)
```

## Arguments

- query:

  Query sentence with mask token.

- model, models:

  Model name(s).

- targets:

  Target words to fill in the mask. Defaults to `NULL` (return the top 5
  most likely words).

- topn:

  Number of the most likely predictions to return. Defaults to `5`.

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

## Value

A data.table of raw results.

## Functions

- `fill_mask()`: Check performance of one model.

- `fill_mask_check()`: Check performance of multiple models.

## Examples

``` r
if (FALSE) { # \dontrun{
query = "Paris is the [MASK] of France."
models = c("bert-base-uncased", "bert-base-cased")

d.check = fill_mask_check(query, models, topn=2)
} # }
```
