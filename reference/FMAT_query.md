# Prepare a data.table of queries and variables for the FMAT.

Prepare a data.table of queries and variables for the FMAT.

## Usage

``` r
FMAT_query(
  query = "Text with [MASK], optionally with {TARGET} and/or {ATTRIB}.",
  MASK = .(),
  TARGET = .(),
  ATTRIB = .()
)
```

## Arguments

- query:

  Query text (should be a character string/vector with at least one
  `[MASK]` token). Multiple queries share the same set of `MASK`,
  `TARGET`, and `ATTRIB`. For multiple queries with different `MASK`,
  `TARGET`, and/or `ATTRIB`, please use
  [`FMAT_query_bind()`](https://psychbruce.github.io/FMAT/reference/FMAT_query_bind.md)
  to combine them.

- MASK:

  A named list of `[MASK]` target words. Must be single words in the
  vocabulary of a certain masked language model.

  - For model vocabulary, see, e.g.,
    <https://huggingface.co/bert-base-uncased/raw/main/vocab.txt>

  - Infrequent words may be not included in a model's vocabulary, and in
    this case you may insert the words into the context by specifying
    either `TARGET` or `ATTRIB`.

- TARGET, ATTRIB:

  A named list of Target/Attribute words or phrases. If specified, then
  `query` must contain `{TARGET}` and/or `{ATTRIB}` (in all uppercase
  and in braces) to be replaced by the words/phrases.

## Value

A data.table of queries and variables.

## See also

[`FMAT_query_bind()`](https://psychbruce.github.io/FMAT/reference/FMAT_query_bind.md)

[`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)

## Examples

``` r
FMAT_query("[MASK] is a nurse.", MASK = .(Male="He", Female="She"))
#>                 query   MASK M_pair M_word
#>                <fctr> <fctr> <fctr> <fctr>
#> 1: [MASK] is a nurse.   Male      1     He
#> 2: [MASK] is a nurse. Female      1    She

FMAT_query(
  c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
  MASK = .(Male="He", Female="She"),
  TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
)
#>        qid                     query   MASK M_pair M_word     TARGET
#>     <fctr>                    <fctr> <fctr> <fctr> <fctr>     <fctr>
#>  1:      1       [MASK] is {TARGET}.   Male      1     He Occupation
#>  2:      1       [MASK] is {TARGET}. Female      1    She Occupation
#>  3:      1       [MASK] is {TARGET}.   Male      1     He Occupation
#>  4:      1       [MASK] is {TARGET}. Female      1    She Occupation
#>  5:      1       [MASK] is {TARGET}.   Male      1     He Occupation
#>  6:      1       [MASK] is {TARGET}. Female      1    She Occupation
#>  7:      2 [MASK] works as {TARGET}.   Male      1     He Occupation
#>  8:      2 [MASK] works as {TARGET}. Female      1    She Occupation
#>  9:      2 [MASK] works as {TARGET}.   Male      1     He Occupation
#> 10:      2 [MASK] works as {TARGET}. Female      1    She Occupation
#> 11:      2 [MASK] works as {TARGET}.   Male      1     He Occupation
#> 12:      2 [MASK] works as {TARGET}. Female      1    She Occupation
#>           T_pair    T_word
#>           <fctr>    <fctr>
#>  1: Occupation.1  a doctor
#>  2: Occupation.1  a doctor
#>  3: Occupation.2   a nurse
#>  4: Occupation.2   a nurse
#>  5: Occupation.3 an artist
#>  6: Occupation.3 an artist
#>  7: Occupation.1  a doctor
#>  8: Occupation.1  a doctor
#>  9: Occupation.2   a nurse
#> 10: Occupation.2   a nurse
#> 11: Occupation.3 an artist
#> 12: Occupation.3 an artist

FMAT_query(
  "The [MASK] {ATTRIB}.",
  MASK = .(Male=c("man", "boy"),
           Female=c("woman", "girl")),
  ATTRIB = .(Masc=c("is masculine", "has a masculine personality"),
             Femi=c("is feminine", "has a feminine personality"))
)
#>                    query   MASK M_pair M_word ATTRIB      A_pair
#>                   <fctr> <fctr> <fctr> <fctr> <fctr>      <fctr>
#>  1: The [MASK] {ATTRIB}.   Male      1    man   Masc Masc-Femi.1
#>  2: The [MASK] {ATTRIB}.   Male      2    boy   Masc Masc-Femi.1
#>  3: The [MASK] {ATTRIB}. Female      1  woman   Masc Masc-Femi.1
#>  4: The [MASK] {ATTRIB}. Female      2   girl   Masc Masc-Femi.1
#>  5: The [MASK] {ATTRIB}.   Male      1    man   Masc Masc-Femi.2
#>  6: The [MASK] {ATTRIB}.   Male      2    boy   Masc Masc-Femi.2
#>  7: The [MASK] {ATTRIB}. Female      1  woman   Masc Masc-Femi.2
#>  8: The [MASK] {ATTRIB}. Female      2   girl   Masc Masc-Femi.2
#>  9: The [MASK] {ATTRIB}.   Male      1    man   Femi Masc-Femi.1
#> 10: The [MASK] {ATTRIB}.   Male      2    boy   Femi Masc-Femi.1
#> 11: The [MASK] {ATTRIB}. Female      1  woman   Femi Masc-Femi.1
#> 12: The [MASK] {ATTRIB}. Female      2   girl   Femi Masc-Femi.1
#> 13: The [MASK] {ATTRIB}.   Male      1    man   Femi Masc-Femi.2
#> 14: The [MASK] {ATTRIB}.   Male      2    boy   Femi Masc-Femi.2
#> 15: The [MASK] {ATTRIB}. Female      1  woman   Femi Masc-Femi.2
#> 16: The [MASK] {ATTRIB}. Female      2   girl   Femi Masc-Femi.2
#>                          A_word
#>                          <fctr>
#>  1:                is masculine
#>  2:                is masculine
#>  3:                is masculine
#>  4:                is masculine
#>  5: has a masculine personality
#>  6: has a masculine personality
#>  7: has a masculine personality
#>  8: has a masculine personality
#>  9:                 is feminine
#> 10:                 is feminine
#> 11:                 is feminine
#> 12:                 is feminine
#> 13:  has a feminine personality
#> 14:  has a feminine personality
#> 15:  has a feminine personality
#> 16:  has a feminine personality
```
