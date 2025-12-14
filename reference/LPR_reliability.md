# Reliability analysis (Cronbach's \\\alpha\\) of LPR.

Reliability analysis (Cronbach's \\\alpha\\) of LPR.

## Usage

``` r
LPR_reliability(fmat, item = c("query", "T_word", "A_word"), by = NULL)
```

## Arguments

- fmat:

  A data.table returned from
  [`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md).

- item:

  Reliability of multiple `"query"` (default), `"T_word"`, or
  `"A_word"`.

- by:

  Variable(s) to split data by. Options can be `"model"`, `"TARGET"`,
  `"ATTRIB"`, or any combination of them.

## Value

A data.table of Cronbach's \\\alpha\\.
