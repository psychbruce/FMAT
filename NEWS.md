**Please check the [latest news (change log)](https://psychbruce.github.io/FMAT/news/index.html) and keep this package updated.**

# FMAT 2024.3

-   The FMAT methodology paper has been accepted (March 14, 2024) for publication in the *Journal of Personality and Social Psychology* (<https://doi.org/10.1037/pspa0000396>)!
-   Added `BERT_download()` (downloading models to local cache folder "%USERPROFILE%/.cache/huggingface") to differentiate from `FMAT_load()` (loading saved models from local cache). But indeed `FMAT_load()` can also download models *silently* if they have not been downloaded.
-   Added `gpu` parameter for `FMAT_load()` to allow for specifying an NVIDIA GPU device on which the fill-mask pipeline will be allocated. GPU may perform 3x faster than CPU for the fill-mask pipeline.
-   Added a summary of device information when using `BERT_download()` and `FMAT_load()`.

# FMAT 2023.8

-   CRAN package publication.
-   Fixed bugs and improved functions.
-   Provided more examples.
-   Now use "YYYY.M" as package version number.

# FMAT 0.0.9 (May 2023)

-   Initial public release on [GitHub](https://github.com/psychbruce/FMAT).

# FMAT 0.0.1 (Jan 2023)

-   Designed basic functions.
