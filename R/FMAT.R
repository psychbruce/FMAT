#### Initialize ####


#' @keywords internal
"_PACKAGE"


#' @import stringr
#' @import data.table
#' @importFrom dplyr left_join mutate
#' @importFrom forcats as_factor
#' @importFrom rvest read_html html_elements html_attr
#' @importFrom stats na.omit
#' @importFrom crayon italic underline green blue magenta
.onAttach = function(libname, pkgname) {
  Sys.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING" = "1")
  Sys.setenv("TF_ENABLE_ONEDNN_OPTS" = "0")
  Sys.setenv("KMP_DUPLICATE_LIB_OK" = "TRUE")
  Sys.setenv("OMP_NUM_THREADS" = "1")
  # Fixed "R Session Aborted" issue on MacOS
  # https://github.com/psychbruce/FMAT/issues/1

  inst.ver = as.character(utils::packageVersion("FMAT"))
  pkgs = c("data.table", "stringr", "forcats")
  suppressMessages({
    suppressWarnings({
      loaded = sapply(pkgs, require, character.only=TRUE)
    })
  })
  packageStartupMessage(
    glue::glue_col("

    {magenta FMAT (v{inst.ver})}
    {blue The Fill-Mask Association Test}

    {magenta Packages also loaded:}
    {green \u2714 data.table, stringr, forcats}

    {magenta Online documentation:}
    {underline https://psychbruce.github.io/FMAT}

    {magenta To use this package in publications, please cite:}
    Bao, H. W. S. (2023). {italic FMAT: The Fill-Mask Association Test} (Version {inst.ver}) [Computer software]. {underline https://doi.org/10.32614/CRAN.package.FMAT}

    Bao, H. W. S. (2024). The Fill-Mask Association Test (FMAT): Measuring propositions in natural language. {italic Journal of Personality and Social Psychology, 127}(3), 537-561. {underline https://doi.org/10.1037/pspa0000396}

    "))
}


#### Utils ####


## #' @importFrom PsychWordVec cc
## #' @export
## PsychWordVec::cc


#' A simple function equivalent to `list`.
#'
#' @param ... Named objects (usually character vectors for this package).
#'
#' @return
#' A list of named objects.
#'
#' @examples
#' .(Male=c("he", "his"), Female=c("she", "her"))
#'
#' @keywords internal
#' @export
. = function(...) list(...)


dtime = function(t0) {
  diff = as.numeric(difftime(Sys.time(), t0, units="secs"))
  if(diff < 1) {
    paste0(round(diff * 1000), "ms")
  } else if(diff < 60) {
    paste0(round(diff, 1), "s")
  } else {
    mins = floor(diff / 60)
    secs = round(diff - mins * 60, 1)
    paste0(mins, "m ", secs, "s")
  }
}


gpu_to_device = function(gpu) {
  cuda = reticulate::import("torch")$cuda$is_available()
  if(missing(gpu))
    gpu = cuda
  if(is.logical(gpu))
    device = ifelse(gpu, 0L, -1L)
  if(is.numeric(gpu))
    device = as.integer(device)
  if(is.character(gpu))
    device = gpu
  if(!device %in% c(-1L, "cpu") & !cuda)
    stop("
      NVIDIA GPU CUDA is not enabled!
      For guidance, see https://psychbruce.github.io/FMAT/",
      call.=FALSE)
  return(device)
}


transformers_init = function(print.info=TRUE) {
  t0 = Sys.time()
  FMAT.ver = as.character(utils::packageVersion("FMAT"))
  reticulate.ver = as.character(utils::packageVersion("reticulate"))

  # os = reticulate::import("os")
  # os$environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
  # os$environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  # Sys.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING" = "1")
  # Sys.setenv("TF_ENABLE_ONEDNN_OPTS" = "0")

  # "R Session Aborted" issue on MacOS
  # https://github.com/psychbruce/FMAT/issues/1
  # os$environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  # os$environ["OMP_NUM_THREADS"] = "1"
  # Sys.setenv("KMP_DUPLICATE_LIB_OK" = "TRUE")
  # Sys.setenv("OMP_NUM_THREADS" = "1")

  torch = reticulate::import("torch")
  torch.ver = torch$`__version__`
  torch.cuda = torch$cuda$is_available()
  if(torch.cuda) {
    # cuda.ver = torch$cuda_version
    # new torch: AttributeError: module 'torch' has no attribute 'cuda_version'
    gpu.info = paste("GPU (Device):", paste(torch$cuda$get_device_name(), collapse=", "))
  } else {
    # cuda.ver = "NULL"
    gpu.info = "To use GPU, see https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration"
  }

  transformers = reticulate::import("transformers")
  tf.ver = transformers$`__version__`

  hf = reticulate::import("huggingface_hub")
  hfh.ver = hf$`__version__`

  # urllib = reticulate::import("urllib3")
  # url.ver = urllib$`__version__`

  if(print.info) {
    cli::cli_alert_info(cli::col_blue("Device Info:

    R Packages:
    FMAT          {FMAT.ver}
    reticulate    {reticulate.ver}

    Python Packages:
    transformers  {tf.ver}
    torch         {torch.ver}
    huggingface-hub  {hfh.ver}

    NVIDIA GPU CUDA Support:
    CUDA Enabled: {torch.cuda}
    {gpu.info}

    (transformers initialized in {dtime(t0)})
    "))
  }

  return(transformers)
}


fill_mask_init = function(transformers, model, device=-1L) {
  cache.folder = get_cache_folder(transformers)
  model.local = get_cached_model_path(cache.folder, model)
  reticulate::py_capture_output({
    config = transformers$AutoConfig$from_pretrained(
      model.local,
      local_files_only = TRUE)
    fill_mask = transformers$pipeline(
      "fill-mask",
      model = model.local,
      config = config,
      model_kwargs = list(local_files_only=TRUE),
      device = device)
  })
  return(fill_mask)
}


#' Compute a vector of weights with a decay rate.
#'
#' @param vector Vector of sequence.
#' @param decay Decay factor for computing weights. A smaller decay value would give greater weight to the former items than to the latter items. The i-th item has raw weight = decay ^ i.
#' - decay = 1: all items are **equally** important
#' - 0 < decay < 1: **first** items are more important
#' - decay > 1: **last** items are more important
#'
#' @return
#' *Normalized* weights (i.e., sum of weights = 1).
#'
#' @seealso
#' [FMAT_run()]
#'
#' @examples
#' # "individualism"
#' weight_decay(c("individual", "##ism"), 0.5)
#' weight_decay(c("individual", "##ism"), 0.8)
#' weight_decay(c("individual", "##ism"), 1)
#' weight_decay(c("individual", "##ism"), 2)
#'
#' # "East Asian people"
#' weight_decay(c("East", "Asian", "people"), 0.5)
#' weight_decay(c("East", "Asian", "people"), 0.8)
#' weight_decay(c("East", "Asian", "people"), 1)
#' weight_decay(c("East", "Asian", "people"), 2)
#'
#' @export
weight_decay = function(vector, decay) {
  weights = decay^seq_along(vector)
  weights = weights / sum(weights)
  names(weights) = vector
  return(weights)
}


add_tokens = function(
    fill_mask,
    tokens,
    weight.decay = 1,
    # = 1: equally important
    # < 1: first more important
    # > 1: last more important
    device = -1L,  # CPU
    verbose.out = TRUE
) {
  # encode new tokens from subwords
  method = ifelse(
    weight.decay==1,
    "mean",
    paste0("mean_weighted_[decay=", weight.decay, "]")
  )
  # debug shortcut:
  # fill_mask = FMAT:::fill_mask_init(FMAT:::transformers_init(), "roberta-base")
  # fill_mask = FMAT:::fill_mask_init(FMAT:::transformers_init(), "albert-base-v1")
  # sapply(encode, function(id) vocab[vocab==id])
  # fill_mask("<mask> people are good.", top_k=5L)
  # fill_mask("<mask> people are good.", targets="African")[[1]]
  # fill_mask("The <mask> people are good.", targets=" African")[[1]]
  # fill_mask("Most <mask> people are good.", targets=" African")[[1]]
  vocab = fill_mask$tokenizer$get_vocab()
  embed = fill_mask$model$get_input_embeddings()$weight$data
  tlist = lapply(tokens, function(token) {
    encode = fill_mask$tokenizer$encode(token, add_special_tokens=FALSE)
    decode = fill_mask$tokenizer$decode(encode)
    if(length(encode)==1) {
      out = FALSE
      token.embed = embed[encode]
    } else {
      out = TRUE
      embeds = embed[encode]  # tensor of subwords
      if(weight.decay==1) {
        token.embed = embeds$mean(0L)
      } else {
        torch = reticulate::import("torch")
        weights = weight_decay(encode, weight.decay)
        weights = torch$tensor(weights, device=device)$view(-1L, 1L)
        embeds.weighted = embeds * weights
        token.embed = embeds.weighted$sum(0L)
      }
    }
    return(list(
      out = out,
      token = token,
      encode = encode,
      decode = decode,
      token.raw = sapply(encode, function(id) vocab[vocab==id]),
      token.embed = token.embed
    ))
  })
  names(tlist) = tokens

  # add new tokens to the tokenizer vocabulary
  new.tokens = as.character(unlist(sapply(
    tlist, function(t) if(t$out) return(t$token)
  )))
  # new.tokens = str_replace(new.tokens, "^ ", "\u0120")
  n.tokens.added = fill_mask$tokenizer$add_tokens(new.tokens)
  if(n.tokens.added > 0)
    cli::cli_alert_success("Added {n.tokens.added} new tokens (weight decay = {weight.decay})")

  # initialize random embeddings for the new tokens
  fill_mask$model$resize_token_embeddings(length(fill_mask$tokenizer))

  # reset new embeddings to (sum or mean) subword token embeddings
  vocab.new = fill_mask$tokenizer$get_vocab()
  embed.new = fill_mask$model$get_input_embeddings()$weight$data
  for(t in tlist) {
    if(t$out) {
      embed.new[vocab.new[[t$token]]] = t$token.embed
      subwords = paste(names(t$token.raw), collapse=", ")
      if(verbose.out)
        cli::cli_alert_success("Added token {.val {t$token}} := embed_{method}({subwords})")
    }
  }

  return(fill_mask)
}


#' Set (change) HuggingFace cache folder temporarily.
#'
#' This function allows you to change the default cache directory (when it lacks storage space) to another path (e.g., your portable SSD) *temporarily*.
#'
#' # Keep in Mind
#'
#' This function takes effect only for the current R session *temporarily*, so you should run this each time BEFORE you use other FMAT functions in an R session.
#'
#' @param path Folder path to store HuggingFace models. If `NULL`, then return the current cache folder.
#'
#' @examples
#' \dontrun{
#' library(FMAT)
#' set_cache_folder("D:/huggingface_cache/")
#' # -> models would be saved to "D:/huggingface_cache/hub/"
#' # run this function each time before using FMAT functions
#'
#' BERT_download()
#' BERT_info()
#' }
#'
#' @export
set_cache_folder = function(path=NULL) {
  if(!is.null(path)) {
    if(!dir.exists(path)) dir.create(path)
    if(!dir.exists(path)) stop("No such directory.", call.=FALSE)
    # os = reticulate::import("os")
    # os$environ["HF_HOME"] = path
    Sys.setenv("HF_HOME" = path)
  }

  transformers = transformers_init(print.info=FALSE)
  cache.folder = get_cache_folder(transformers)

  if(!is.null(path)) {
    if(dirname(cache.folder) != str_remove(path, "/$")) {
      cli::cli_alert_danger("Cannot change cache folder in this R session!")
      stop("Please restart R and run `set_cache_folder()` before other FMAT functions!", call.=FALSE)
    }

    cli::cli_alert_success("Changed HuggingFace cache folder temporarily to {.path {path}}")
    cli::cli_alert_success("Models would be downloaded or could be moved to {.path {paste0(path, 'hub/')}}")
  } else {
    Sys.setenv("HF_HOME" = cache.folder)
    cli::cli_alert_success("Current HuggingFace cache folder is {.path {cache.folder}}")
  }
}


get_cache_folder = function(transformers) {
  str_replace_all(transformers$TRANSFORMERS_CACHE, "\\\\", "/")
}


get_cached_models = function(cache.folder) {
  models.name = list.files(cache.folder, "^models--")
  if(length(models.name) > 0) {
    dm = rbindlist(lapply(paste0(cache.folder, "/", models.name), function(folder) {
      models.file = list.files(folder, pattern="(model.safetensors$|pytorch_model.bin$|tf_model.h5$)", recursive=TRUE, full.names=TRUE)
      # file.size = paste(paste0(sprintf("%.0f", file.size(models.file) / 1024^2), " MB"), collapse=" / ")
      file.size.MB = round(file.size(models.file[1]) / 1024^2)
      download.date = paste(str_remove(file.mtime(models.file), " .*"), collapse=" / ")
      return(data.table(model=NA, file.size.MB, download.date))
    }))
    dm$model = str_replace_all(str_remove(models.name, "^models--"), "--", "/")
  } else {
    dm = NULL
  }
  return(dm)
}


get_cached_model_path = function(cache.folder, model) {
  model.folder = model_folder(cache.folder, model)
  model.path = list.files(model.folder, pattern="(model.safetensors$|pytorch_model.bin$|tf_model.h5$)", recursive=TRUE, full.names=TRUE)[1]
  return(dirname(model.path))
}


model_folder = function(cache.folder, model) {
  paste0(cache.folder, "/models--", str_replace_all(model, "/", "--"))
}


check_models_downloaded = function(local.models, models) {
  if(length(base::setdiff(models, local.models) > 0)) {
    cli::cli_alert_danger("{.val {models}} not found in local cache folder")
    stop("Please check model names or first use `BERT_download()` to download models!", call.=FALSE)
  }
}


#### BERT ####


#' Download and save BERT models to local cache folder.
#'
#' Download and save BERT models to local cache folder "%USERPROFILE%/.cache/huggingface".
#'
#' @param models A character vector of model names at
#' [HuggingFace](https://huggingface.co/models).
#' @param verbose Alert if a model has been downloaded.
#' Defaults to `FALSE`.
#'
#' @return
#' Invisibly return a data.table of basic file information of local models.
#'
#' @seealso
#' [set_cache_folder()]
#'
#' [BERT_info()]
#'
#' [BERT_vocab()]
#'
#' @examples
#' \dontrun{
#' models = c("bert-base-uncased", "bert-base-cased")
#' BERT_download(models)
#'
#' BERT_download()  # check downloaded models
#'
#' BERT_info()  # information of all downloaded models
#' }
#'
#' @export
BERT_download = function(models=NULL, verbose=FALSE) {
  transformers = transformers_init(print.info=!is.null(models))
  cache.folder = get_cache_folder(transformers)

  # if(mirror) {
  #   os = reticulate::import("os")
  #   os$environ["HF_INFERENCE_ENDPOINT"] = "https://hf-mirror.com"
  #   Sys.setenv("HF_INFERENCE_ENDPOINT" = "https://hf-mirror.com")
  #   # default: "https://api-inference.huggingface.com"
  # }

  if(!is.null(models)) {
    lapply(as.character(models), function(model) {
      model.path = get_cached_model_path(cache.folder, model)
      if(is.na(model.path)) {
        model.folder = model_folder(cache.folder, model)
        unlink(model.folder, recursive=TRUE)
        success = FALSE
        try({
          cli::cli_h1("Downloading model {.val {model}}")
          cli::cli_alert("(1) Downloading configuration...")
          transformers$AutoConfig$from_pretrained(model)
          cli::cli_alert("(2) Downloading tokenizer...")
          transformers$AutoTokenizer$from_pretrained(model)
          cli::cli_alert("(3) Downloading model...")
          transformers$AutoModel$from_pretrained(model)
          cli::cli_alert_success("Successfully downloaded model {.val {model}}")
          gc()
          success = TRUE
        })
        if(!success) unlink(model.folder, recursive=TRUE)
      } else {
        if(verbose)
          cli::cli_alert_success("Model has been downloaded: {.val {model}}")
      }
    })
  }

  cache.sizegb = sum(file.size(list.files(cache.folder, recursive=TRUE, full.names=TRUE))) / 1024^3
  local.models = get_cached_models(cache.folder)

  if(is.null(local.models)) {
    cli::cli_alert_warning("No models in {.path {cache.folder}}.")
  } else {
    cli::cli_alert_success(paste(
      "Downloaded {.val {nrow(local.models)}} models",
      "saved in {.path {cache.folder}}",
      "({sprintf('%.2f', cache.sizegb)} GB)"))
  }

  invisible(local.models)
}


#' Remove BERT models from local cache folder.
#'
#' @param models Model names.
#'
#' @return
#' `NULL`.
#'
#' @export
BERT_remove = function(models) {
  transformers = transformers_init(print.info=FALSE)
  cache.folder = get_cache_folder(transformers)
  lapply(as.character(models), function(model) {
    model.folder = model_folder(cache.folder, model)
    if(dir.exists(model.folder)) {
      unlink(model.folder, recursive=TRUE)
      cli::cli_alert_success("Model removed: {.val {model}}")
    } else {
      cli::cli_alert_danger("Model not found: {.val {model}}")
    }
  })
  invisible(NULL)
}


#' Get basic information of BERT models.
#'
#' @inheritParams BERT_download
#'
#' @return
#' A data.table:
#' - model name
#' - model type
#' - number of parameters
#' - vocabulary size (of input token embeddings)
#' - embedding dimensions (of input token embeddings)
#' - hidden layers
#' - attention heads
#' - \[MASK\] token
#'
#' @seealso
#' [BERT_download()]
#'
#' [BERT_vocab()]
#'
#' @examples
#' \dontrun{
#' models = c("bert-base-uncased", "bert-base-cased")
#' BERT_info(models)
#'
#' BERT_info()  # information of all downloaded models
#' # speed: ~1.2s/model for first use; <1s afterwards
#' }
#'
#' @export
BERT_info = function(models=NULL) {
  transformers = transformers_init(print.info=FALSE)
  cache.folder = get_cache_folder(transformers)
  infos.folder = paste0(cache.folder, "/.info/")
  if(!dir.exists(infos.folder)) dir.create(infos.folder)
  local.models = get_cached_models(cache.folder)
  if(is.null(models)) models = local.models$model
  models = as.character(models)
  check_models_downloaded(local.models$model, models)
  dm = data.table()

  op = options()
  options(cli.progress_bar_style="bar")
  cli::cli_progress_bar(
    clear = FALSE,
    total = length(models),
    format = paste(
      "{cli::pb_spin} Reading model info",
      "{cli::pb_current}/{cli::pb_total}",
      "{cli::pb_bar} {cli::pb_percent}",
      "[{cli::pb_elapsed_clock}]"),
    format_done = paste(
      "{cli::col_green(cli::symbol$tick)}",
      "{cli::pb_total} models info read in {cli::pb_elapsed}")
  )

  for(model in models) {
    model.info.file = paste0(infos.folder, str_replace_all(model, "/", "--"), ".rda")
    if(!file.exists(model.info.file)) {
      # cli::cli_progress_step("Loading {.val {model}}")
      model.local = get_cached_model_path(cache.folder, model)
      try({
        reticulate::py_capture_output({
          tokenizer = transformers$AutoTokenizer$from_pretrained(
            model.local, local_files_only=TRUE)
          model.obj = transformers$AutoModel$from_pretrained(
            model.local, local_files_only=TRUE)
        })
        vocab = embed = NA
        # word.embeddings = model.obj$embeddings$word_embeddings$weight$data$shape
        # vocab = word.embeddings[0]
        # embed = word.embeddings[1]
        word.embeddings = model.obj$get_input_embeddings()
        vocab = word.embeddings$num_embeddings
        embed = word.embeddings$embedding_dim
        di = data.table(
          model = as.factor(model),
          type = as.factor(model.obj$config$model_type),
          param = model.obj$num_parameters(),
          vocab = vocab,
          embed = embed,
          layer = model.obj$config$num_hidden_layers,
          heads = model.obj$config$num_attention_heads,
          mask = as.factor(tokenizer$mask_token)
        )
        save(di, file=model.info.file)
        rm(di, tokenizer, model.obj, word.embeddings)
        gc()
      })
    }
    load(model.info.file)
    dm = rbind(dm, di)
    if(nrow(di)==0) {
      cli::cli_alert_danger("Model {.val {model}} may not have BERT-like config")
    }
    cli::cli_progress_update()
  }

  cli::cli_progress_done()
  options(op)

  dm$model = factor(dm$model, levels=models)

  return(dm)
}


#' Scrape the initial commit date of BERT models.
#'
#' @inheritParams BERT_info
#'
#' @return
#' A data.table:
#' - model name
#' - initial commit date (scraped from huggingface commit history)
#'
#' @examples
#' \dontrun{
#' model.date = BERT_info_date()
#' # get all models from cache folder
#'
#' one.model.date = FMAT:::get_model_date("bert-base-uncased")
#' # call the internal function to scrape a model
#' # that may not have been saved in cache folder
#' }
#'
#' @export
BERT_info_date = function(models=NULL) {
  transformers = transformers_init(print.info=FALSE)
  cache.folder = get_cache_folder(transformers)
  dates.folder = paste0(cache.folder, "/.date/")
  if(!dir.exists(dates.folder)) dir.create(dates.folder)
  if(is.null(models)) {
    models = str_replace_all(str_remove(list.files(cache.folder, "^models--"), "^models--"), "--", "/")
  }
  models = as.character(models)
  dd = data.table()

  op = options()
  options(cli.progress_bar_style = "bar")
  cli::cli_progress_bar("Scraping model date:", total=length(models), clear=TRUE)

  for(model in models) {
    model.date.file = paste0(dates.folder, str_replace_all(model, "/", "--"), ".txt")
    if(!file.exists(model.date.file)) {
      try({
        dates = get_model_date(model)  # sorted dates
        writeLines(dates, model.date.file)
      })
    }
    dd = rbind(
      dd,
      data.table(
        model = as.factor(model),
        date = readLines(model.date.file)[1]
      )
    )
    cli::cli_progress_update()
  }

  cli::cli_progress_done()
  options(op)

  return(dd)
}


get_model_date = function(model) {
  url = paste0("https://huggingface.co/", model, "/commits/main")
  xml = read_html(url)
  dates = html_attr(html_elements(xml, "time"), "datetime")
  return(sort(str_sub(dates, 1, 10)))
}


#' Check if mask words are in the model vocabulary.
#'
#' @inheritParams BERT_download
#' @param mask.words Option words filling in the mask.
#' @param add.tokens Add new tokens (for out-of-vocabulary words or phrases) to model vocabulary? Defaults to `FALSE`.
#' - Default method of producing the new token embeddings is computing the (equally weighted) average subword token embeddings. To change the weights of different subwords, specify `weight.decay`.
#' - It just adds tokens temporarily without changing the raw model file.
#' @param add.verbose Print subwords of each new token? Defaults to `FALSE`.
#' @param weight.decay Decay factor of relative importance of multiple subwords. Defaults to `1` (see [weight_decay()] for computational details). A smaller decay value would give greater weight to the former subwords than to the latter subwords. The i-th subword has raw weight = decay ^ i.
#' - decay = 1: all subwords are **equally** important (default)
#' - 0 < decay < 1: **first** subwords are more important
#' - decay > 1: **last** subwords are more important
#'
#' For example, decay = 0.5 would give 0.5 and 0.25 (with normalized weights 0.667 and 0.333) to two subwords (e.g., "individualism" = 0.667 "individual" + 0.333 "##ism").
#'
#' @return
#' A data.table of model name, mask word, real token (replaced if out of vocabulary), and token id (0~N).
#'
#' @seealso
#' [BERT_download()]
#'
#' [BERT_info()]
#'
#' [FMAT_run()]
#'
#' @examples
#' \dontrun{
#' models = c("bert-base-uncased", "bert-base-cased")
#' BERT_info(models)
#'
#' BERT_vocab(models, c("bruce", "Bruce"))
#'
#' BERT_vocab(models, 2020:2025)  # some are out-of-vocabulary
#' BERT_vocab(models, 2020:2025, add.tokens=TRUE)  # add vocab
#'
#' BERT_vocab(models,
#'            c("individualism", "artificial intelligence"),
#'            add.tokens=TRUE)
#' }
#'
#' @export
BERT_vocab = function(
    models, mask.words,
    add.tokens = FALSE,
    add.verbose = FALSE,
    weight.decay = 1
) {
  transformers = transformers_init(print.info=FALSE)
  mask.words = as.character(mask.words)

  maps = rbindlist(lapply(as.character(models), function(model) {
    try({
      map = NULL
      if(add.tokens) {
        cli::cli_alert("{.val {model}} add tokens...")
      }
      fill_mask = fill_mask_init(transformers, model)
      id.max = max(unlist(fill_mask$tokenizer$get_vocab()))
      if(add.tokens) {
        fill_mask = add_tokens(
          fill_mask,
          mask.words,
          weight.decay = weight.decay,
          verbose.out = add.verbose)
      }
      vocab = fill_mask$tokenizer$get_vocab()
      map = rbindlist(lapply(mask.words, function(mask) {
        encode = fill_mask$tokenizer$encode(mask, add_special_tokens=FALSE)
        if(length(encode) > 1 & max(encode) <= id.max) {
          id = encode[1]
          token = paste(names(vocab[vocab==id]), "(out-of-vocabulary)")
        } else {
          id = as.integer(fill_mask$get_target_ids(mask))
          token = names(vocab[vocab==id])
        }
        data.table(model=as_factor(model), M_word=as_factor(mask), token=token, token.id=id)
      }))
    })
    return(map)
  }))

  warning_oov(maps)

  return(maps)
}


#### FMAT ####


fix_pair = function(X, var="MASK") {
  ns = sapply(X, length)
  if(length(unique(ns)) > 1) {
    cli::cli_alert_warning("Unequal number of items in {var}.")
    nmax = max(ns)
    for(i in 1:length(X)) {
      if(ns[i] < nmax)
        X[[i]] = c(X[[i]], rep(NA, nmax - ns[i]))
    }
  }
  return(X)
}


# query = "[MASK] is ABC."
# expand_pair(query, .(High=c("high", "strong"), Low=c("low", "weak")))
# expand_pair(query, .(H="high", M="medium", L="low"))
# X = .(Flower=c("rose", "iris", "lily"), Pos=c("health", "happiness", "love", "peace"))
# expand_full(query, X)


expand_pair = function(query, X, var="MASK") {
  d = data.frame(X)
  dt = rbindlist(lapply(names(d), function(var) {
    data.table(query = query,
               var = var,
               pair = seq_len(nrow(d)),
               word = d[[var]])
  }))
  dt$var = as_factor(dt$var)
  dt$pair = as_factor(dt$pair)
  dt$word = as_factor(dt$word)
  v = str_sub(var, 1, 1)
  names(dt) = c("query", var, paste0(v, "_pair"), paste0(v, "_word"))
  return(dt)
}


expand_full = function(query, X) {
  d = expand.grid(X)
  dt = data.table(query = factor(query),
                  TARGET = factor(names(d)[1]),
                  T_word = as_factor(d[[1]]),
                  ATTRIB = factor(names(d)[2]),
                  A_word = as_factor(d[[2]]))
  return(dt)
}


map_query = function(.x, .f, ...) {
  dq = purrr::map_df(.x, .f, ...)  # .x should be query (chr vec)
  dq$query = as_factor(dq$query)
  return(dq)
}


append_X = function(dq, X, var="TARGET") {
  n = nrow(dq)
  prefix = paste(names(X), collapse="-")

  dx = data.frame(X)
  dx = rbindlist(lapply(names(dx), function(x) {
    data.table(x = rep(x, each=n),
               pair = rep(paste0(prefix, ".", seq_len(nrow(dx))), each=n),
               word = rep(dx[[x]], each=n))
  }))
  dx$x = as_factor(dx$x)
  dx$pair = as_factor(dx$pair)
  dx$word = as_factor(dx$word)
  v = str_sub(var, 1, 1)
  names(dx) = c(var, paste0(v, "_pair"), paste0(v, "_word"))

  return(cbind(dq, dx))
}


#' Prepare a data.table of queries and variables for the FMAT.
#'
#' @param query Query text (should be a character string/vector with at least one `[MASK]` token). Multiple queries share the same set of `MASK`, `TARGET`, and `ATTRIB`. For multiple queries with different `MASK`, `TARGET`, and/or `ATTRIB`, please use [FMAT_query_bind()] to combine them.
#' @param MASK A named list of `[MASK]` target words. Must be single words in the vocabulary of a certain masked language model.
#' - For model vocabulary, see, e.g., <https://huggingface.co/bert-base-uncased/raw/main/vocab.txt>
#' - Infrequent words may be not included in a model's vocabulary, and in this case you may insert the words into the context by specifying either `TARGET` or `ATTRIB`.
#' @param TARGET,ATTRIB A named list of Target/Attribute words or phrases. If specified, then `query` must contain `{TARGET}` and/or `{ATTRIB}` (in all uppercase and in braces) to be replaced by the words/phrases.
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' [FMAT_query_bind()]
#'
#' [FMAT_run()]
#'
#' @examples
#' \donttest{FMAT_query("[MASK] is a nurse.", MASK = .(Male="He", Female="She"))
#'
#' FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
#' )
#'
#' FMAT_query(
#'   "The [MASK] {ATTRIB}.",
#'   MASK = .(Male=c("man", "boy"),
#'            Female=c("woman", "girl")),
#'   ATTRIB = .(Masc=c("is masculine", "has a masculine personality"),
#'              Femi=c("is feminine", "has a feminine personality"))
#' )
#' }
#' @export
FMAT_query = function(
    query = "Text with [MASK], optionally with {TARGET} and/or {ATTRIB}.",
    MASK = .(),
    TARGET = .(),
    ATTRIB = .()
) {
  if(any(str_count(query, "\\[MASK\\]") == 0L))
    stop("`query` should contain a [MASK] token!", call.=FALSE)
  if(any(str_count(query, "\\[MASK\\]") >= 2L))
    stop("`query` should contain only *one* [MASK] token!", call.=FALSE)
  if(length(MASK) == 0) {
    stop("Please specify `MASK` (the targets of [MASK])!", call.=FALSE)
  } else {
    MASK = fix_pair(MASK)
  }

  sapply(MASK, function(x) {
    x = na.omit(x)
    if(anyDuplicated(x)) {
      dup = x[duplicated(x)]
      cli::cli_alert_danger("Duplicated mask words: {.val {unique(dup)}}")
      stop("Duplicated mask words found in `MASK`!", call.=FALSE)
    }
  })

  if(length(TARGET) == 0 & length(ATTRIB) == 0) {
    # No TARGET or ATTRIB
    type = "M"
    dq = map_query(query, expand_pair, MASK)
  } else if(length(TARGET) > 0 & length(ATTRIB) == 0) {
    # Only TARGET
    type = "MT"
    dq = append_X(map_query(query, expand_pair, MASK),
                  TARGET, "TARGET")
  } else if(length(TARGET) == 0 & length(ATTRIB) > 0) {
    # Only ATTRIB
    type = "MA"
    dq = append_X(map_query(query, expand_pair, MASK),
                  ATTRIB, "ATTRIB")
  } else if(length(TARGET) > 0 & length(ATTRIB) > 0) {
    # Both TARGET and ATTRIB
    type = "MTA"
    dm = map_query(query, expand_pair, MASK)
    dx = map_query(query, function(q, target, attrib) {
      # rbind(
      #   expand_full(q, c(target[1], attrib[1])),
      #   expand_full(q, c(target[1], attrib[2])),
      #   expand_full(q, c(target[2], attrib[1])),
      #   expand_full(q, c(target[2], attrib[2]))
      # )
      rbindlist(
        lapply(seq_along(target), function(i) {
          rbindlist(
            lapply(seq_along(attrib), function(j) {
              expand_full(q, c(target[i], attrib[j]))
            })
          )
        })
      )
    }, TARGET, ATTRIB)
    dq = unique(plyr::adply(dx, 1, function(x) cbind(dm, x)))
  }

  if(length(query) > 1)
    dq = cbind(data.table(qid = as.factor(as.numeric(dq$query))), dq)
  attr(dq, "type") = type
  return(na.omit(dq[order(query)]))
}


#' Combine multiple query data.tables and renumber query ids.
#'
#' @param ... Query data.tables returned from [FMAT_query()].
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' [FMAT_query()]
#'
#' [FMAT_run()]
#'
#' @examples
#' \donttest{FMAT_query_bind(
#'   FMAT_query(
#'     "[MASK] is {TARGET}.",
#'     MASK = .(Male="He", Female="She"),
#'     TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
#'   ),
#'   FMAT_query(
#'     "[MASK] occupation is {TARGET}.",
#'     MASK = .(Male="His", Female="Her"),
#'     TARGET = .(Occupation=c("doctor", "nurse", "artist"))
#'   )
#' )
#' }
#' @export
FMAT_query_bind = function(...) {
  types = sapply(list(...), attr, "type")
  type = unique(types)
  if(length(type) > 1)
    stop("Queries should have the same structure.", call.=FALSE)

  dqs = rbind(...)

  if("qid" %in% names(dqs)) dqs$qid = NULL
  dqs = cbind(data.table(qid = as.factor(as.numeric(dqs$query))), dqs)

  attr(dqs, "type") = type
  return(dqs)
}


#' Run the fill-mask pipeline and check the raw results.
#'
#' This function is only for technical check. Please use [FMAT_run()] for general purposes.
#'
#' @describeIn fill_mask Check performance of one model.
#'
#' @inheritParams FMAT_run
#' @param query Query sentence with mask token.
#' @param model,models Model name(s).
#' @param targets Target words to fill in the mask.
#' Defaults to `NULL` (return the top 5 most likely words).
#' @param topn Number of the most likely predictions to return. Defaults to `5`.
#'
#' @return
#' A data.table of raw results.
#'
#' @examples
#' \dontrun{
#' query = "Paris is the [MASK] of France."
#' models = c("bert-base-uncased", "bert-base-cased")
#'
#' d.check = fill_mask_check(query, models, topn=2)
#' }
#'
#' @export
fill_mask = function(query, model, targets=NULL, topn=5, gpu) {
  if(length(model)>1)
    stop("Please use `fill_mask_check()` for multiple models.", call.=FALSE)

  device = gpu_to_device(gpu)
  if(!is.null(targets)) {
    targets = as.character(unique(targets))
    topn = length(targets)
  }
  topn = as.integer(topn)

  transformers = reticulate::import("transformers")
  cli::cli_alert("Loading {.val {model}}")
  fill_mask = fill_mask_init(transformers, model, device)
  mask.token = fill_mask$tokenizer$mask_token
  if(mask.token!="[MASK]")
    query = str_replace_all(query, "\\[MASK\\]", mask.token)

  res = fill_mask(inputs=query, targets=targets, top_k=topn)

  d = cbind(data.table(model=as.factor(model)),
            rbindlist(res))
  rm(fill_mask)
  gc()
  return(d)
}


#' @inheritParams fill_mask
#' @describeIn fill_mask Check performance of multiple models.
#'
#' @export
fill_mask_check = function(query, models, targets=NULL, topn=5, gpu) {
  op = options()
  options(cli.progress_bar_style="bar")
  cli::cli_progress_bar(
    clear = FALSE,
    total = length(models),
    format = paste(
      "{cli::pb_spin} Checking model performance",
      "({cli::pb_current}/{cli::pb_total})",
      "{cli::pb_bar} {cli::pb_percent}"),
    format_done = paste(
      "{cli::col_green(cli::symbol$tick)}",
      "{cli::pb_total} models checked in {cli::pb_elapsed}")
  )

  dt = data.table()
  for(model in as.character(models)) {
    try({
      t0 = Sys.time()
      di = fill_mask(query, model, targets, topn, gpu)
      dt = rbind(dt, di)
      cli::cli_alert("------- ({dtime(t0)})")
    })
    cli::cli_progress_update()
  }
  cli::cli_progress_done()
  options(op)
  return(dt)
}


#' Specify models that require special treatment to ensure accuracy.
#'
#' @param uncased Regular expression pattern (matching model names) for uncased models.
#' @param u2581,u0120 Regular expression pattern (matching model names) for models that require a special prefix character when performing whole-word fill-mask pipeline.
#'
#' **WARNING**: The developer is unable to check all models, so users need to check the models they use and modify these parameters if necessary.
#' - `u2581`: add prefix `\u2581` (white space) for all mask words
#' - `u0120`: add prefix `\u0120` (white space) for only non-starting mask words
#' @param u2581.excl,u0120.excl Exclusions to negate `u2581` and `u0120` matching results.
#'
#' @return
#' A list of regular expression patterns.
#'
#' @seealso
#' [FMAT_run()]
#'
#' @examples
#' special_case()
#'
#' @export
special_case = function(
    uncased = "uncased|albert|electra|muhtasham",
    u2581 = "albert|xlm-roberta|xlnet",
    u2581.excl = "chinese",
    u0120 = "roberta|bart|deberta|bertweet-large|ModernBERT",
    u0120.excl = "chinese|xlm-|kornosk/"
) {
  list(
    uncased = uncased,
    prefix.u2581 = u2581,
    prefix.u2581.excl = u2581.excl,
    prefix.u0120 = u0120,
    prefix.u0120.excl = u0120.excl
  )
}


#' Run the fill-mask pipeline on multiple models (CPU / GPU).
#'
#' Run the fill-mask pipeline on multiple models with CPU or GPU (faster but requires an NVIDIA GPU device).
#'
#' The function automatically adjusts for the compatibility of tokens used in certain models: (1) for uncased models (e.g., ALBERT), it turns tokens to lowercase; (2) for models that use `<mask>` rather than `[MASK]`, it automatically uses the corrected mask token; (3) for models that require a prefix to estimate whole words than subwords (e.g., ALBERT, RoBERTa), it adds a white space before each mask option word. See [special_case()] for details.
#'
#' These changes only affect the `token` variable in the returned data, but will not affect the `M_word` variable. Thus, users may analyze data based on the unchanged `M_word` rather than the `token`.
#'
#' Note also that there may be extremely trivial differences (after 5~6 significant digits) in the raw probability estimates between using CPU and GPU, but these differences would have little impact on main results.
#'
#' @inheritParams BERT_vocab
#' @param data A data.table returned from [FMAT_query()] or [FMAT_query_bind()].
#' @param gpu Use GPU (3x faster than CPU) to run the fill-mask pipeline? Defaults to missing value that will *automatically* use available GPU (if not available, then use CPU). An NVIDIA GPU device (e.g., GeForce RTX Series) is required to use GPU. See [Guidance for GPU Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration).
#'
#' Options passing on to the `device` parameter in Python:
#' - `FALSE`: CPU (`device = -1`).
#' - `TRUE`: GPU (`device = 0`).
#' - Others: passing on to [`transformers.pipeline(device=...)`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.device) which defines the device (e.g., `"cpu"`, `"cuda:0"`, or a GPU device id like `1`) on which the pipeline will be allocated.
#' @param pattern.special See [special_case()] for details.
#' @param file File name of `.RData` to save the returned data.
#' @param progress Show a progress bar? Defaults to `TRUE`.
#' @param warning Alert warning of out-of-vocabulary word(s)? Defaults to `TRUE`.
#' @param na.out Replace probabilities of out-of-vocabulary word(s) with `NA`? Defaults to `TRUE`.
#'
#' @return
#' A data.table (class `fmat`) appending `data` with these new variables:
#' - `model`: model name.
#' - `output`: complete sentence output with unmasked token.
#' - `token`: actual token to be filled in the blank mask
#'   (a note "out-of-vocabulary" will be added
#'   if the original word is not found in the model vocabulary).
#' - `prob`: (raw) conditional probability of the unmasked token given the provided context, estimated by the masked language model.
#'   - Raw probabilities should *NOT* be directly used or interpreted. Please use [summary.fmat()] to *contrast* between a pair of probabilities.
#'
#' @seealso
#' [set_cache_folder()]
#'
#' [BERT_download()]
#'
#' [BERT_vocab()]
#'
#' [FMAT_query()]
#'
#' [FMAT_query_bind()]
#'
#' [summary.fmat()]
#'
#' [special_case()]
#'
#' [weight_decay()]
#'
#' @examples
#' ## Running the examples requires the models downloaded
#'
#' \dontrun{
#' models = c("bert-base-uncased", "bert-base-cased")
#'
#' query1 = FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=c("a doctor", "a nurse", "an artist"))
#' )
#' data1 = FMAT_run(models, query1)
#' summary(data1, target.pair=FALSE)
#'
#' query2 = FMAT_query(
#'   "The [MASK] {ATTRIB}.",
#'   MASK = .(Male=c("man", "boy"),
#'            Female=c("woman", "girl")),
#'   ATTRIB = .(Masc=c("is masculine", "has a masculine personality"),
#'              Femi=c("is feminine", "has a feminine personality"))
#' )
#' data2 = FMAT_run(models, query2)
#' summary(data2, mask.pair=FALSE)
#' summary(data2)
#' }
#'
#' @export
FMAT_run = function(
    models,
    data,
    gpu,
    add.tokens = FALSE,
    add.verbose = FALSE,
    weight.decay = 1,
    pattern.special = special_case(),
    file = NULL,
    progress = TRUE,
    warning = TRUE,
    na.out = TRUE
) {
  t0 = Sys.time()
  n.models = length(models)
  type = attr(data, "type")
  device = gpu_to_device(gpu)
  device.info = ifelse(device %in% c(-1L, "cpu"), "CPU", "GPU")
  progress = ifelse(progress, "text", "none")

  transformers = transformers_init()
  cache.folder = get_cache_folder(transformers)
  cat("\n")
  cli::cli_text("Loading models from {.path {cache.folder}} ...")
  local.models = get_cached_models(cache.folder)
  check_models_downloaded(local.models$model, models)

  query = .query = mask = .mask = M_word = T_word = A_word = token = NULL

  #### .query (final query sentences) ####
  data = mutate(data, .query = str_replace(query, "\\[mask\\]", "[MASK]"))
  if("TARGET" %in% names(data))
    data = mutate(data, .query = str_replace(.query, "\\{TARGET\\}", as.character(T_word)))
  if("ATTRIB" %in% names(data))
    data = mutate(data, .query = str_replace(.query, "\\{ATTRIB\\}", as.character(A_word)))
  n.unique.query = length(unique(data$.query))
  cli::cli_text("Task on {device.info}: {n.models} models * {n.unique.query} unique queries (type: {.val {type}})")
  cat(" ")

  #### function for one model ####
  onerun = function(model, data) {
    ## ---- One Run Begin ---- ##
    model.id = paste0(which(model==models), "/", n.models)
    cli::cli_h1("[{model.id}] {.val {model}}")

    # BERT model special cases
    uncased = str_detect(model, pattern.special$uncased)
    prefix.u2581 = str_detect(
      model, pattern.special$prefix.u2581) &
      !str_detect(model, pattern.special$prefix.u2581.excl)
    prefix.u0120 = str_detect(
      model, pattern.special$prefix.u0120) &
      !str_detect(model, pattern.special$prefix.u0120.excl)

    #### .mask (final mask target words) ####
    data = mutate(data, .mask = as.character(M_word))
    if(uncased) {
      cli::cli_alert_success("Info: uncased model")
      data = mutate(data, .mask = tolower(.mask))
    }
    if(prefix.u2581) {
      # albert
      cli::cli_alert_success("Info: \\u2581 used as whole-word prefix")
      data = mutate(data, .mask = paste0("\u2581", .mask))
    }
    if(prefix.u0120) {
      # roberta
      cli::cli_alert_success("Info: \\u0120 used as whole-word prefix")
      data = mutate(data, .mask = ifelse(
        str_detect(.query, "^\\[MASK\\]"),
        .mask,
        paste0(" ", .mask)))
      # "\u0120" -> " "
      # more accurate for filling whole words! (2025.12.19)
    }

    #### fill_mask ####
    fill_mask = fill_mask_init(transformers, model, device)
    mask.token = fill_mask$tokenizer$mask_token
    id.max = max(unlist(fill_mask$tokenizer$get_vocab()))
    if(mask.token!="[MASK]") {
      cli::cli_alert_success("Info: {mask.token} used as the [MASK] token")
      data = mutate(data, .query = str_replace(.query, "\\[MASK\\]", mask.token))
    }

    #### add tokens for out-of-vocabulary words ####
    if(add.tokens) {
      fill_mask = add_tokens(
        fill_mask,
        unique(data$.mask),
        weight.decay = weight.decay,
        device = device,
        verbose.out = add.verbose)
    }

    #### unmask (list version) ####
    unmask = function(fill_mask, d, mask.list) {
      out = reticulate::py_capture_output({
        res = fill_mask(
          inputs = d$.query,
          targets = mask.list,
          top_k = as.integer(length(mask.list)))
      })
      d = rbindlist(res)[, c("token", "sequence", "score")]
      names(d) = c("token.id", "output", "prob")
      return(d)
    }

    #### mapping mask token id ####
    mask_id_map = function(fill_mask, mask.options) {
      vocab = fill_mask$tokenizer$get_vocab()
      map = rbindlist(lapply(mask.options, function(mask) {
        encode = fill_mask$tokenizer$encode(mask, add_special_tokens=FALSE)
        if(length(encode) > 1 & max(encode) <= id.max) {
          # out of vocabulary
          id = encode[1]
          token = paste(names(vocab[vocab==id]), "(out-of-vocabulary)")
        } else {
          # in vocabulary:
          # -  length(encode) = 1
          # -  length(encode) > 1 & max(encode) > id.max
          # (
          #  bert-base-uncased:
          #  after add_tokens("omani"),
          #  encode("romanian") => r + omani + an
          #  where "omani" is larger than id.max
          # )
          id = as.integer(fill_mask$get_target_ids(mask))
          token = names(vocab[vocab==id])
        }
        data.table(.mask=mask, token.id=id, token=token)
      }))
      return(map)
    }

    #### progress running ####
    map = mask_id_map(fill_mask, unique(data$.mask))
    t1 = Sys.time()
    dq = plyr::adply(unique(data[, ".query"]), 1,
                     unmask,
                     fill_mask=fill_mask,
                     mask.list=map$.mask,
                     .progress=progress)
    dq = left_join(dq, map, by="token.id")  # important `by` var
    data = left_join(data, dq[, c(".query", ".mask", "output", "token", "prob")],
                     by=c(".query", ".mask"))
    rm(dq)
    data$.query = data$.mask = NULL
    data = cbind(data.table(model=as_factor(model)), data)

    if(min(data$prob, na.rm=TRUE)==0) {
      n.zero.prob = sum(data$prob==0, na.rm=TRUE)
      cli::cli_alert_danger("{n.zero.prob} zero probability estimates!")
    }

    mins = as.numeric(difftime(Sys.time(), t1, units="mins"))
    speed = sprintf("%.0f", n.unique.query / mins)
    cat(paste0("  (", dtime(t1), ") [",
               speed, " unique queries/min]",
               "\n"))
    rm(fill_mask)
    gc()

    return(data)
    ## ---- One Run End ---- ##
  }

  #### run ####
  data = rbindlist(lapply(as.character(models), onerun, data=data))
  cat("\n")
  attr(data, "type") = type
  class(data) = c("fmat", class(data))
  gc()
  cli::cli_alert_success("Done ({dtime(t0)})")

  if(warning) warning_oov(data)
  if(na.out) data[str_detect(token, "out-of-vocabulary")]$prob = NA

  if(!is.null(file)) {
    if(!str_detect(file, "\\.[Rr][Dd]a(ta)?$"))
      file = paste0(file, ".RData")
    save(data, file=file)
    cli::cli_alert_success("Data saved to {.val {file}}")
  }

  invisible(data)
}


warning_oov = function(data) {
  d.oov = unique(data[str_detect(data$token, "out-of-vocabulary"),
                      c("M_word", "token")])
  d.oov$token = str_remove(d.oov$token, " \\(out-of-vocabulary\\)")
  if(nrow(d.oov) > 0) {
    oovs = unique(d.oov$M_word)
    for(oov in oovs) {
      di = d.oov[d.oov$M_word==oov]
      cli::cli_alert_warning("
      Replaced out-of-vocabulary word {.val {oov}} by: {.val {di$token}}")
    }
  }
}


#' \[S3 method\] Summarize the results for the FMAT.
#'
#' Summarize the results of *Log Probability Ratio* (LPR), which indicates the *relative* (vs. *absolute*) association between concepts.
#'
#' The LPR of just one contrast (e.g., only between a pair of attributes) may *not* be sufficient for a proper interpretation of the results, and may further require a second contrast (e.g., between a pair of targets).
#'
#' Users are suggested to use linear mixed models (with the R packages `nlme` or `lme4`/`lmerTest`) to perform the formal analyses and hypothesis tests based on the LPR.
#'
#' @inheritParams FMAT_run
#' @param object A data.table (class `fmat`) returned from [FMAT_run()].
#' @param mask.pair,target.pair,attrib.pair Pairwise contrast of `[MASK]`, `TARGET`, `ATTRIB`? Defaults to `TRUE`.
#' @param ... Other arguments (currently not used).
#'
#' @return
#' A data.table of the summarized results with Log Probability Ratio (LPR).
#'
#' @seealso
#' [FMAT_run()]
#'
#' @examples
#' # see examples in `FMAT_run`
#'
#' @export
summary.fmat = function(
    object,
    mask.pair = TRUE,
    target.pair = TRUE,
    attrib.pair = TRUE,
    warning = TRUE,
    ...) {
  if(warning) warning_oov(object)
  type = attr(object, "type")

  M_word = T_word = A_word = MASK = TARGET = ATTRIB = prob = LPR = NULL
  T_pair = T_pair_i = T_pair_j = A_pair = A_pair_i = A_pair_j = NULL

  if(mask.pair) {
    gvars = c("model", "query", "M_pair",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word")
    grouping.vars = intersect(names(object), gvars)
    dt = object[, .(
      MASK = paste(MASK, collapse=" - "),
      M_word = paste(M_word, collapse=" - "),
      LPR = log(prob[1]) - log(prob[2])
    ), by = grouping.vars]
    dt$MASK = as_factor(dt$MASK)
    dt$M_word = as_factor(dt$M_word)
    dt$M_pair = NULL
  } else {
    dvars = c("model", "query", "MASK", "M_word",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word",
              "prob")
    dt.vars = intersect(names(object), dvars)
    dt = object[, dt.vars, with=FALSE]
    dt$LPR = log(dt$prob)
    dt$prob = NULL
  }

  if(type=="MT") {
    if(target.pair) {
      dt = dt[, .(
        TARGET = paste(TARGET, collapse=" - "),
        T_word = paste(T_word, collapse=" - "),
        LPR = LPR[1] - LPR[2]
      ), by = c("model", "query", "MASK", "M_word", "T_pair")]
      dt$TARGET = as_factor(dt$TARGET)
      dt$T_word = as_factor(dt$T_word)
      dt$T_pair = NULL
    }
  }

  if(type=="MA") {
    if(attrib.pair) {
      dt = dt[, .(
        ATTRIB = paste(ATTRIB, collapse=" - "),
        A_word = paste(A_word, collapse=" - "),
        LPR = LPR[1] - LPR[2]
      ), by = c("model", "query", "MASK", "M_word", "A_pair")]
      dt$ATTRIB = as_factor(dt$ATTRIB)
      dt$A_word = as_factor(dt$A_word)
      dt$A_pair = NULL
    }
  }

  if(type=="MTA") {
    if(target.pair) {
      dt[, T_pair_i := (as.numeric(TARGET)+1) %/% 2]
      dt[, T_pair_j := 1:.N,
         by=c("model", "query", "MASK", "M_word",
              "ATTRIB", "A_word", "TARGET")]
      dt[, T_pair := as_factor(paste(T_pair_i, T_pair_j))]
      dt = dt[, .(
        TARGET = paste(TARGET, collapse=" - "),
        T_word = paste(T_word, collapse=" - "),
        LPR = LPR[1] - LPR[2]
      ), by = c("model", "query", "MASK", "M_word",
                "ATTRIB", "A_word", "T_pair")]
      dt$TARGET = as_factor(dt$TARGET)
      dt$T_word = as_factor(dt$T_word)
      dt$T_pair = NULL
    }
    if(attrib.pair) {
      dt[, A_pair_i := (as.numeric(ATTRIB)+1) %/% 2]
      dt[, A_pair_j := 1:.N,
         by=c("model", "query", "MASK", "M_word",
              "TARGET", "T_word", "ATTRIB")]
      dt[, A_pair := as_factor(paste(A_pair_i, A_pair_j))]
      dt = dt[, .(
        ATTRIB = paste(ATTRIB, collapse=" - "),
        A_word = paste(A_word, collapse=" - "),
        LPR = LPR[1] - LPR[2]
      ), by = c("model", "query", "MASK", "M_word",
                "TARGET", "T_word", "A_pair")]
      dt$ATTRIB = as_factor(dt$ATTRIB)
      dt$A_word = as_factor(dt$A_word)
      dt$A_pair = NULL
    }
  }

  return(dt)
}


#' Intraclass correlation coefficient (ICC) of BERT models.
#'
#' Interrater agreement of *log probabilities* (treated as "ratings"/rows) among BERT language models (treated as "raters"/columns), with both row and column as ("two-way") random effects.
#'
#' @param data Raw data returned from [FMAT_run()] (with variable `prob`) or its summarized data obtained with [summary.fmat()] (with variable `LPR`).
#' @param type Interrater `"agreement"` (default) or `"consistency"`.
#' @param unit Reliability of `"average"` scores (default) or `"single"` scores.
#'
#' @return
#' A data.frame of ICC.
#'
#' @export
ICC_models = function(data, type="agreement", unit="average") {
  data = as.data.frame(data)
  data$ID = rowidv(data, cols="model")
  data$model = as.numeric(data$model)
  if("prob" %in% names(data)) {
    data$logp = log(data$prob)
    var = "Log Probability (Raw)"
  } else if("LPR" %in% names(data)) {
    data$logp = data$LPR
    var = "Log Probability Ratio (LPR)"
  } else {
    stop("`data` should be obtained with `FMAT_run()`!", call.=FALSE)
  }
  d = tidyr::pivot_wider(
    data[, c("ID", "model", "logp")],
    names_from="model",
    values_from="logp",
    names_prefix="logp")[, -1]
  ICC = irr::icc(d, model="twoway", unit=unit, type=type)
  res = data.frame(items = ICC$subjects,
                   raters = ICC$raters,
                   ICC = ICC$value)
  names(res)[3] = paste0("ICC.", unit)
  row.names(res) = var
  return(res)
}


#' Reliability analysis (Cronbach's \eqn{\alpha}) of LPR.
#'
#' @param fmat A data.table returned from [summary.fmat()].
#' @param item Reliability of multiple `"query"` (default), `"T_word"`, or `"A_word"`.
#' @param by Variable(s) to split data by. Options can be `"model"`, `"TARGET"`, `"ATTRIB"`, or any combination of them.
#'
#' @return
#' A data.frame of Cronbach's \eqn{\alpha}.
#'
#' @export
LPR_reliability = function(
    fmat,
    item = c("query", "T_word", "A_word"),
    by = NULL
) {
  item = match.arg(item)
  alphas = plyr::ddply(fmat, by, function(x) {
    x = as.data.frame(x)
    x[[item]] = as.numeric(x[[item]])
    if("T_pair" %in% names(x)) x$T_pair = NULL
    if("A_pair" %in% names(x)) x$A_pair = NULL
    x = tidyr::pivot_wider(
      x,
      names_from = item,
      names_glue = paste0("LPR.{", item, "}"),
      values_from = "LPR")
    suppressWarnings({
      suppressMessages({
        alpha = psych::alpha(
          dplyr::select(x, tidyr::starts_with("LPR")),
          delete=FALSE,
          warnings=FALSE)
      })
    })
    data.frame(n.obs = nrow(x),
               k.items = alpha$nvar,
               alpha = alpha$total$raw_alpha)
  })
  if(is.null(by)) {
    alphas[[1]] = NULL
    row.names(alphas) = item
  }
  return(alphas)
}

