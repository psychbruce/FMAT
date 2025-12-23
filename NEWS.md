**Please check the [latest news (change log)](https://psychbruce.github.io/FMAT/news/index.html) and keep this package updated.**

# FMAT 2026.1

This version brings crucial changes and improvements to the `add.tokens` method of `FMAT_run()`. Old versions are completely deprecated.

-   Deprecated the `add.method` parameter of `FMAT_run()`: Now it always computes **average** subword token embeddings, with another new parameter `weight.decay` (default value = 1, i.e., equally weighted) that can specify the relative importance of multiple subwords.
    -   A smaller decay value would give greater weight to the former subwords than to the latter subwords.
    -   The *i*-th subword $w_i$ has the raw weight score as $\text{weight}(w_i) = decay^i$.
    -   See also `weight_decay()` for computational details.
-   Added `special_case()` as an explicit function to better specify models requiring special treatment, which was previously just a parameter of `FMAT_run()`.
-   Improved the internal technical treatment for subword token combination when "\\u0120" is the special prefix. This is important for more accurate out-of-vocabulary fill-mask probability estimates of whole words and phrases, mainly relevant to models using the RoBERTa architecture and only influencing results under `add.tokens=TRUE`. Note that `FMAT_run()` results with `add.tokens=FALSE` are not affected by this change.
-   Improved progress output information of `FMAT_run()`.
-   Improved `ICC_models()` to support ICC estimates of both log probability (raw) and log probability ratio (LPR).

# FMAT 2025.12

-   Changed the default `add.method` of `add.tokens` from `"sum"` to `"mean"`, relevant to `BERT_vocab()` and `FMAT_run()`. Using the averaged rather than the summed subword token embeddings for out-of-vocabulary tokens would have a smaller impact on the probability estimates of vocabulary tokens.
-   Improved functionality for the latest versions of Python packages.
-   Refined help pages in the style of Roxygen markdown.

# FMAT 2025.4

-   Added `BERT_remove()`: Remove models from local cache folder.
-   Added `fill_mask()` and `fill_mask_check()`: These functions are only for technical check (i.e., checking the raw results of fill-mask pipeline). Normal users should usually use `FMAT_run()`.
-   Added `pattern.special` argument for `FMAT_run()`: Regular expression patterns (matching model names) for special model cases that are uncased or require a special prefix character in certain situations.
    -   **WARNING**: As the developer is not able to check all models, users are responsible for checking the models they would use and for modifying this argument if necessary.
        -   `prefix.u2581`: adding prefix ⁠`\u2581⁠` for all mask words
        -   `prefix.u0120`: adding prefix ⁠`\u0120`⁠ for only non-starting mask words
-   Improved `set_cache_folder()`, `BERT_download()`, `BERT_info()`, and `BERT_info_date()`.
    -   Now model information read from model objects `BERT_info()` and model initial commit date scraped from HuggingFace `BERT_info_date()` will be saved in subfolders of local cache: `/.info/` and `/.date/`, respectively.
-   Deprecated `FMAT_load()`.
-   Fixed "R Session Aborted" issue on MacOS (see #1).
-   Set necessary environment variables automatically when `library(FMAT)`:
    -   `Sys.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING" = "1")`
    -   `Sys.setenv("TF_ENABLE_ONEDNN_OPTS" = "0")`
    -   `Sys.setenv("KMP_DUPLICATE_LIB_OK" = "TRUE")`
    -   `Sys.setenv("OMP_NUM_THREADS" = "1")`

# FMAT 2025.3

-   Added `set_cache_folder()`: Set (change) HuggingFace cache folder temporarily.
    -   **Keep in mind**: This function takes effect only for the current R session temporarily, so you should run this each time *before* you use other FMAT functions in an R session.
-   Added `BERT_info_date()`: Scrape the initial commit date of BERT models from HuggingFace.
-   Improved `BERT_download()` and `BERT_info()`.
-   Updated the formal citation format of the *JPSP* article.

# FMAT 2024.7

-   Added the DOI link for the online published *JPSP* article: <https://doi.org/10.1037/pspa0000396>.

# FMAT 2024.6

-   Fixed bugs: Now only `BERT_download()` connects to the Internet, while all the other functions run in an offline way.
-   Improved installation guidance for Python packages.

# FMAT 2024.5

-   Added `BERT_info()`.
-   Added `add.tokens` and `add.method` arguments for `BERT_vocab()` and `FMAT_run()`: An *experimental* functionality to add new tokens (e.g., out-of-vocabulary words, compound words, or even phrases) as [MASK] options. Validation is still needed for this novel practice (one of my ongoing projects), so currently please only use at your own risk, waiting until the publication of my validation work.
-   All functions except `BERT_download()` now import local model files only, without automatically downloading models. Users must first use `BERT_download()` to download models.
-   Deprecating `FMAT_load()`: Better to use `FMAT_run()` directly.

# FMAT 2024.4

-   Added `BERT_vocab()` and `ICC_models()`.
-   Improved `summary.fmat()`, `FMAT_query()`, and `FMAT_run()` (significantly faster because now it can *simultaneously* estimate all [MASK] options for each unique query sentence, with running time only depending on the number of unique queries but not on the number of [MASK] options).
-   If you use the `reticulate` package version ≥ 1.36.1, then `FMAT` should be updated to ≥ 2024.4. Otherwise, out-of-vocabulary [MASK] words may not be identified and marked. Now `FMAT_run()` directly uses model vocabulary and token ID to match [MASK] words. To check if a [MASK] word is in the model vocabulary, please use `BERT_vocab()`.

# FMAT 2024.3

-   The FMAT methodology paper has been accepted (March 14, 2024) for publication in the *Journal of Personality and Social Psychology: Attitudes and Social Cognition* (DOI: 10.1037/pspa0000396)!
-   Added `BERT_download()` (downloading models to local cache folder "%USERPROFILE%/.cache/huggingface") to differentiate from `FMAT_load()` (loading saved models from local cache). But indeed `FMAT_load()` can also download models *silently* if they have not been downloaded.
-   Added `gpu` argument (see [Guidance for GPU Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration)) in `FMAT_run()` to allow for specifying an NVIDIA GPU device on which the fill-mask pipeline will be allocated. GPU roughly performs 3x faster than CPU for the fill-mask pipeline. By default, `FMAT_run()` would automatically detect and use any available GPU with an installed CUDA-supported Python `torch` package (if not, it would use CPU).
-   Added running speed information (queries/min) for `FMAT_run()`.
-   Added device information for `BERT_download()`, `FMAT_load()`, and `FMAT_run()`.
-   Deprecated `parallel` in `FMAT_run()`: `FMAT_run(model.names, data, gpu=TRUE)` is the fastest.
-   A progress bar is displayed by default for `progress` in `FMAT_run()`.

# FMAT 2023.8

-   CRAN package publication.
-   Fixed bugs and improved functions.
-   Provided more examples.
-   Now use "YYYY.M" as package version number.

# FMAT 0.0.9 (May 2023)

-   Initial public release on [GitHub](https://github.com/psychbruce/FMAT).

# FMAT 0.0.1 (Jan 2023)

-   Designed basic functions.
