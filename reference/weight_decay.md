# Compute a vector of weights with a decay rate.

Compute a vector of weights with a decay rate.

## Usage

``` r
weight_decay(vector, decay)
```

## Arguments

- vector:

  Vector of sequence.

- decay:

  Decay factor for computing weights. A smaller decay value would give
  greater weight to the former items than to the latter items. The i-th
  item has raw weight = decay ^ i.

  - decay = 1: all items are **equally** important

  - 0 \< decay \< 1: **first** items are more important

  - decay \> 1: **last** items are more important

## Value

*Normalized* weights (i.e., sum of weights = 1).

## See also

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
# "individualism"
weight_decay(c("individual", "##ism"), 0.5)
#> individual      ##ism 
#>  0.6666667  0.3333333 
weight_decay(c("individual", "##ism"), 0.8)
#> individual      ##ism 
#>  0.5555556  0.4444444 
weight_decay(c("individual", "##ism"), 1)
#> individual      ##ism 
#>        0.5        0.5 
weight_decay(c("individual", "##ism"), 2)
#> individual      ##ism 
#>  0.3333333  0.6666667 

# "East Asian people"
weight_decay(c("East", "Asian", "people"), 0.5)
#>      East     Asian    people 
#> 0.5714286 0.2857143 0.1428571 
weight_decay(c("East", "Asian", "people"), 0.8)
#>      East     Asian    people 
#> 0.4098361 0.3278689 0.2622951 
weight_decay(c("East", "Asian", "people"), 1)
#>      East     Asian    people 
#> 0.3333333 0.3333333 0.3333333 
weight_decay(c("East", "Asian", "people"), 2)
#>      East     Asian    people 
#> 0.1428571 0.2857143 0.5714286 
```
