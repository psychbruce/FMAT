# Changelog

## FMAT 2025.12

- Changed the default `add.method` of `add.tokens` from `"sum"` to
  `"mean"`, relevant to
  [`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)
  and
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).
  Using the averaged rather than the summed subword token embeddings for
  out-of-vocabulary tokens would have a smaller impact on the
  probability estimates of vocabulary tokens.
- Improved functionality for the latest versions of Python packages.
- Refined help pages in the style of Roxygen markdown.

## FMAT 2025.4

CRAN release: 2025-04-08

- Added
  [`BERT_remove()`](https://psychbruce.github.io/FMAT/reference/BERT_remove.md):
  Remove models from local cache folder.
- Added
  [`fill_mask()`](https://psychbruce.github.io/FMAT/reference/fill_mask.md)
  and
  [`fill_mask_check()`](https://psychbruce.github.io/FMAT/reference/fill_mask.md):
  These functions are only for technical check (i.e., checking the raw
  results of fill-mask pipeline). Normal users should usually use
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).
- Added `pattern.special` argument for
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md):
  Regular expression patterns (matching model names) for special model
  cases that are uncased or require a special prefix character in
  certain situations.
  - **WARNING**: As the developer is not able to check all models, users
    are responsible for checking the models they would use and for
    modifying this argument if necessary.
    - `prefix.u2581`: adding prefix ⁠`\u2581⁠` for all mask words
    - `prefix.u0120`: adding prefix ⁠`\u0120`⁠ for only non-starting mask
      words
- Improved
  [`set_cache_folder()`](https://psychbruce.github.io/FMAT/reference/set_cache_folder.md),
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md),
  [`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md),
  and
  [`BERT_info_date()`](https://psychbruce.github.io/FMAT/reference/BERT_info_date.md).
  - Now model information read from model objects
    [`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md)
    and model initial commit date scraped from HuggingFace
    [`BERT_info_date()`](https://psychbruce.github.io/FMAT/reference/BERT_info_date.md)
    will be saved in subfolders of local cache: `/.info/` and `/.date/`,
    respectively.
- Deprecated `FMAT_load()`.
- Fixed “R Session Aborted” issue on MacOS (see
  [\#1](https://github.com/psychbruce/FMAT/issues/1)).
- Set necessary environment variables automatically when
  [`library(FMAT)`](https://psychbruce.github.io/FMAT/):
  - `Sys.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING" = "1")`
  - `Sys.setenv("TF_ENABLE_ONEDNN_OPTS" = "0")`
  - `Sys.setenv("KMP_DUPLICATE_LIB_OK" = "TRUE")`
  - `Sys.setenv("OMP_NUM_THREADS" = "1")`

## FMAT 2025.3

CRAN release: 2025-03-19

- Added
  [`set_cache_folder()`](https://psychbruce.github.io/FMAT/reference/set_cache_folder.md):
  Set (change) HuggingFace cache folder temporarily.
  - **Keep in mind**: This function takes effect only for the current R
    session temporarily, so you should run this each time *before* you
    use other FMAT functions in an R session.
- Added
  [`BERT_info_date()`](https://psychbruce.github.io/FMAT/reference/BERT_info_date.md):
  Scrape the initial commit date of BERT models from HuggingFace.
- Improved
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)
  and
  [`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md).
- Updated the formal citation format of the *JPSP* article.

## FMAT 2024.7

CRAN release: 2024-07-29

- Added the DOI link for the online published *JPSP* article:
  <https://doi.org/10.1037/pspa0000396>.

## FMAT 2024.6

CRAN release: 2024-06-12

- Fixed bugs: Now only
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)
  connects to the Internet, while all the other functions run in an
  offline way.
- Improved installation guidance for Python packages.

## FMAT 2024.5

CRAN release: 2024-05-19

- Added
  [`BERT_info()`](https://psychbruce.github.io/FMAT/reference/BERT_info.md).
- Added `add.tokens` and `add.method` arguments for
  [`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)
  and
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md):
  An *experimental* functionality to add new tokens (e.g.,
  out-of-vocabulary words, compound words, or even phrases) as \[MASK\]
  options. Validation is still needed for this novel practice (one of my
  ongoing projects), so currently please only use at your own risk,
  waiting until the publication of my validation work.
- All functions except
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)
  now import local model files only, without automatically downloading
  models. Users must first use
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)
  to download models.
- Deprecating `FMAT_load()`: Better to use
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  directly.

## FMAT 2024.4

CRAN release: 2024-04-29

- Added
  [`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md)
  and
  [`ICC_models()`](https://psychbruce.github.io/FMAT/reference/ICC_models.md).
- Improved
  [`summary.fmat()`](https://psychbruce.github.io/FMAT/reference/summary.fmat.md),
  [`FMAT_query()`](https://psychbruce.github.io/FMAT/reference/FMAT_query.md),
  and
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  (significantly faster because now it can *simultaneously* estimate all
  \[MASK\] options for each unique query sentence, with running time
  only depending on the number of unique queries but not on the number
  of \[MASK\] options).
- If you use the `reticulate` package version ≥ 1.36.1, then `FMAT`
  should be updated to ≥ 2024.4. Otherwise, out-of-vocabulary \[MASK\]
  words may not be identified and marked. Now
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  directly uses model vocabulary and token ID to match \[MASK\] words.
  To check if a \[MASK\] word is in the model vocabulary, please use
  [`BERT_vocab()`](https://psychbruce.github.io/FMAT/reference/BERT_vocab.md).

## FMAT 2024.3

CRAN release: 2024-03-22

- The FMAT methodology paper has been accepted (March 14, 2024) for
  publication in the *Journal of Personality and Social Psychology:
  Attitudes and Social Cognition* (DOI: 10.1037/pspa0000396)!
- Added
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md)
  (downloading models to local cache folder
  “%USERPROFILE%/.cache/huggingface”) to differentiate from
  `FMAT_load()` (loading saved models from local cache). But indeed
  `FMAT_load()` can also download models *silently* if they have not
  been downloaded.
- Added `gpu` argument (see [Guidance for GPU
  Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration))
  in
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  to allow for specifying an NVIDIA GPU device on which the fill-mask
  pipeline will be allocated. GPU roughly performs 3x faster than CPU
  for the fill-mask pipeline. By default,
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md)
  would automatically detect and use any available GPU with an installed
  CUDA-supported Python `torch` package (if not, it would use CPU).
- Added running speed information (queries/min) for
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).
- Added device information for
  [`BERT_download()`](https://psychbruce.github.io/FMAT/reference/BERT_download.md),
  `FMAT_load()`, and
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).
- Deprecated `parallel` in
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md):
  `FMAT_run(model.names, data, gpu=TRUE)` is the fastest.
- A progress bar is displayed by default for `progress` in
  [`FMAT_run()`](https://psychbruce.github.io/FMAT/reference/FMAT_run.md).

## FMAT 2023.8

CRAN release: 2023-08-11

- CRAN package publication.
- Fixed bugs and improved functions.
- Provided more examples.
- Now use “YYYY.M” as package version number.

## FMAT 0.0.9 (May 2023)

- Initial public release on
  [GitHub](https://github.com/psychbruce/FMAT).

## FMAT 0.0.1 (Jan 2023)

- Designed basic functions.
