# Set (change) HuggingFace cache folder temporarily.

This function allows you to change the default cache directory (when it
lacks storage space) to another path (e.g., your portable SSD)
*temporarily*.

## Usage

``` r
set_cache_folder(path = NULL)
```

## Arguments

- path:

  Folder path to store HuggingFace models. If `NULL`, then return the
  current cache folder.

## Keep in Mind

This function takes effect only for the current R session *temporarily*,
so you should run this each time BEFORE you use other FMAT functions in
an R session.

## Examples

``` r
if (FALSE) { # \dontrun{
library(FMAT)
set_cache_folder("D:/huggingface_cache/")
# -> models would be saved to "D:/huggingface_cache/hub/"
# run this function each time before using FMAT functions

BERT_download()
BERT_info()
} # }
```
