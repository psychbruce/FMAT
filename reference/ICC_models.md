# Intraclass correlation coefficient (ICC) of BERT models.

Interrater agreement of *log probabilities* (treated as "ratings"/rows)
among BERT language models (treated as "raters"/columns), with both row
and column as ("two-way") random effects.

## Usage

``` r
ICC_models(data, type = "agreement", unit = "average")
```

## Arguments

- data:

  Raw data returned from
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  (with variable `prob`) or its summarized data obtained with
  [`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md)
  (with variable `LPR`).

- type:

  Interrater `"agreement"` (default) or `"consistency"`.

- unit:

  Reliability of `"average"` scores (default) or `"single"` scores.

## Value

A data.frame of ICC.
