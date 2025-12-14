# Combine multiple query data.tables and renumber query ids.

Combine multiple query data.tables and renumber query ids.

## Usage

``` r
FMAT_query_bind(...)
```

## Arguments

- ...:

  Query data.tables returned from
  [`FMAT_query()`](https://psychbruce.github.io/FMAT/reference/FMAT_query.md).

## Value

A data.table of queries and variables.

## See also

[`FMAT_query()`](https://psychbruce.github.io/FMAT/reference/FMAT_query.md)

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
FMAT_query_bind(
  FMAT_query(
    "[MASK] is {TARGET}.",
    MASK = .(Male="He", Female="She"),
    TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
  ),
  FMAT_query(
    "[MASK] occupation is {TARGET}.",
    MASK = .(Male="His", Female="Her"),
    TARGET = .(Occupation=c("doctor", "nurse", "artist"))
  )
)
#>        qid                          query   MASK M_pair M_word     TARGET
#>     <fctr>                         <fctr> <fctr> <fctr> <fctr>     <fctr>
#>  1:      1            [MASK] is {TARGET}.   Male      1     He Occupation
#>  2:      1            [MASK] is {TARGET}. Female      1    She Occupation
#>  3:      1            [MASK] is {TARGET}.   Male      1     He Occupation
#>  4:      1            [MASK] is {TARGET}. Female      1    She Occupation
#>  5:      1            [MASK] is {TARGET}.   Male      1     He Occupation
#>  6:      1            [MASK] is {TARGET}. Female      1    She Occupation
#>  7:      2 [MASK] occupation is {TARGET}.   Male      1    His Occupation
#>  8:      2 [MASK] occupation is {TARGET}. Female      1    Her Occupation
#>  9:      2 [MASK] occupation is {TARGET}.   Male      1    His Occupation
#> 10:      2 [MASK] occupation is {TARGET}. Female      1    Her Occupation
#> 11:      2 [MASK] occupation is {TARGET}.   Male      1    His Occupation
#> 12:      2 [MASK] occupation is {TARGET}. Female      1    Her Occupation
#>           T_pair    T_word
#>           <fctr>    <fctr>
#>  1: Occupation.1  a doctor
#>  2: Occupation.1  a doctor
#>  3: Occupation.2   a nurse
#>  4: Occupation.2   a nurse
#>  5: Occupation.3 an artist
#>  6: Occupation.3 an artist
#>  7: Occupation.1    doctor
#>  8: Occupation.1    doctor
#>  9: Occupation.2     nurse
#> 10: Occupation.2     nurse
#> 11: Occupation.3    artist
#> 12: Occupation.3    artist
```
