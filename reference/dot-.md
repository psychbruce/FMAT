# A simple function equivalent to `list`.

A simple function equivalent to `list`.

## Usage

``` r
.(...)
```

## Arguments

- ...:

  Named objects (usually character vectors for this package).

## Value

A list of named objects.

## Examples

``` r
.(Male=c("he", "his"), Female=c("she", "her"))
#> $Male
#> [1] "he"  "his"
#> 
#> $Female
#> [1] "she" "her"
#> 
```
