# Scrape the initial commit date of BERT models.

Scrape the initial commit date of BERT models.

## Usage

``` r
BERT_info_date(models = NULL)
```

## Arguments

- models:

  A character vector of model names at
  [HuggingFace](https://huggingface.co/models).

## Value

A data.table:

- model name

- initial commit date (scraped from huggingface commit history)

## Examples

``` r
if (FALSE) { # \dontrun{
model.date = BERT_info_date()
# get all models from cache folder

one.model.date = FMAT:::get_model_date("bert-base-uncased")
# call the internal function to scrape a model
# that may not have been saved in cache folder
} # }
```
