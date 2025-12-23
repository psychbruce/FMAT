# Specify models that require special treatment to ensure accuracy.

Specify models that require special treatment to ensure accuracy.

## Usage

``` r
special_case(
  uncased = "uncased|albert|electra|muhtasham",
  u2581 = "albert|xlm-roberta|xlnet",
  u2581.excl = "chinese",
  u0120 = "roberta|bart|deberta|bertweet-large|ModernBERT",
  u0120.excl = "chinese|xlm-|kornosk/"
)
```

## Arguments

- uncased:

  Regular expression pattern (matching model names) for uncased models.

- u2581, u0120:

  Regular expression pattern (matching model names) for models that
  require a special prefix character when performing whole-word
  fill-mask pipeline.

  **WARNING**: The developer is unable to check all models, so users
  need to check the models they use and modify these parameters if
  necessary.

  - `u2581`: add prefix `\u2581` (white space) for all mask words

  - `u0120`: add prefix `\u0120` (white space) for only non-starting
    mask words

- u2581.excl, u0120.excl:

  Exclusions to negate `u2581` and `u0120` matching results.

## Value

A list of regular expression patterns.

## See also

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
special_case()
#> $uncased
#> [1] "uncased|albert|electra|muhtasham"
#> 
#> $prefix.u2581
#> [1] "albert|xlm-roberta|xlnet"
#> 
#> $prefix.u2581.excl
#> [1] "chinese"
#> 
#> $prefix.u0120
#> [1] "roberta|bart|deberta|bertweet-large|ModernBERT"
#> 
#> $prefix.u0120.excl
#> [1] "chinese|xlm-|kornosk/"
#> 
```
