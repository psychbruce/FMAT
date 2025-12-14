# \[S3 method\] Summarize the results for the FMAT.

Summarize the results of *Log Probability Ratio* (LPR), which indicates
the *relative* (vs. *absolute*) association between concepts.

## Usage

``` r
# S3 method for class 'fmat'
summary(
  object,
  mask.pair = TRUE,
  target.pair = TRUE,
  attrib.pair = TRUE,
  warning = TRUE,
  ...
)
```

## Arguments

- object:

  A data.table (class `fmat`) returned from
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).

- mask.pair, target.pair, attrib.pair:

  Pairwise contrast of `[MASK]`, `TARGET`, `ATTRIB`? Defaults to `TRUE`.

- warning:

  Alert warning of out-of-vocabulary word(s)? Defaults to `TRUE`.

- ...:

  Other arguments (currently not used).

## Value

A data.table of the summarized results with Log Probability Ratio (LPR).

## Details

The LPR of just one contrast (e.g., only between a pair of attributes)
may *not* be sufficient for a proper interpretation of the results, and
may further require a second contrast (e.g., between a pair of targets).

Users are suggested to use linear mixed models (with the R packages
`nlme` or `lme4`/`lmerTest`) to perform the formal analyses and
hypothesis tests based on the LPR.

## See also

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
# see examples in `FMAT_run`
```
