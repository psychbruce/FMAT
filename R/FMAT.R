#### Initialize ####


#' @import stringr
#' @import data.table
#' @importFrom forcats as_factor
#' @importFrom stats na.omit
.onAttach = function(libname, pkgname) {
  inst.ver = as.character(utils::packageVersion("FMAT"))
  pkg.date = substr(utils::packageDate("FMAT"), 1, 4)
  pkgs = c("data.table", "stringr", "forcats")
  suppressMessages({
    suppressWarnings({
      loaded = sapply(pkgs, require, character.only=TRUE)
    })
  })
  if(all(loaded)) {
    packageStartupMessage(glue::glue_col("

    {magenta FMAT (v{inst.ver})}
    {blue The Fill-Mask Association Test}

    {magenta Packages also loaded:}
    {green \u2714 data.table, stringr, forcats}

    {magenta Online documentation:}
    {underline https://psychbruce.github.io/FMAT}

    {magenta To use this package in publications, please cite:}
    Bao, H.-W.-S. ({pkg.date}). "),
    glue::glue_col("{italic FMAT: The Fill-Mask Association Test}"),
    glue::glue_col(" (Version {inst.ver}) [Computer software]. "),
    glue::glue_col("{underline https://CRAN.R-project.org/package=FMAT}"),
    "\n\n",
    glue::glue_col("Bao, H.-W.-S. (2024). The Fill-Mask Association Test (FMAT): "),
    glue::glue_col("Measuring propositions in natural language. "),
    glue::glue_col("{italic Journal of Personality and Social Psychology. }"),
    glue::glue_col("Advance online publication. {underline https://doi.org/10.1037/pspa0000396}"),
    "\n")
  }
}


#### Basic ####


#' @importFrom PsychWordVec cc
#' @export
PsychWordVec::cc


#' A simple function equivalent to `list`.
#'
#' @param ... Named objects (usually character vectors for this package).
#'
#' @return A list of named objects.
#'
#' @examples
#' .(Male=cc("he, his"), Female=cc("she, her"))
#' list(Male=cc("he, his"), Female=cc("she, her"))  # the same
#'
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


transformers_init = function() {
  reticulate::py_capture_output({
    os = reticulate::import("os")
    os$environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os$environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    torch = reticulate::import("torch")
    torch.ver = torch$`__version__`
    torch.cuda = torch$cuda$is_available()
    if(torch.cuda) {
      cuda.ver = torch$cuda_version
      gpu.info = paste("GPU (Device):", paste(torch$cuda$get_device_name(), collapse=", "))
    } else {
      cuda.ver = "NULL"
      gpu.info = paste("(To use GPU, install PyTorch with CUDA support,",
                       "see https://pytorch.org/get-started)")
    }

    transformers = reticulate::import("transformers")
    tf.ver = transformers$`__version__`
  })
  cli::cli_alert_info(cli::col_blue("Device Info:

  Python Environment:
  Package       Version
  transformers  {tf.ver}
  torch         {torch.ver}

  NVIDIA GPU CUDA Support:
  CUDA Enabled: {torch.cuda}
  CUDA Version: {cuda.ver}
  {gpu.info}
  "))
  return(transformers)
}


find_cached_models = function(cache.folder) {
  models.name = list.files(cache.folder, "^models--")
  if(length(models.name) > 0) {
    models.size = sapply(paste0(cache.folder, "/", models.name), function(folder) {
      models.file = list.files(folder, pattern="(model.safetensors$|pytorch_model.bin$|tf_model.h5$)", recursive=TRUE, full.names=TRUE)
      paste(paste0(sprintf("%.0f", file.size(models.file) / 1024^2), " MB"), collapse=" / ")
    })
    models.name = str_replace_all(str_remove(models.name, "^models--"), "--", "/")
    models.info = data.frame(Size=models.size, row.names=models.name)
  } else {
    models.info = NULL
  }
  return(models.info)
}


#### BERT ####


#' Download and save BERT models to local cache folder.
#'
#' Download and save BERT models to local cache folder "%USERPROFILE%/.cache/huggingface".
#'
#' @param models Model names at
#' [HuggingFace](https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers).
#'
#' @return
#' No return value.
#'
#' @seealso
#' [`FMAT_load`]
#'
#' @examples
#' \dontrun{
#' model.names = c("bert-base-uncased", "bert-base-cased")
#' BERT_download(model.names)
#'
#' BERT_download()  # check downloaded models
#' }
#'
#' @export
BERT_download = function(models=NULL) {
  transformers = transformers_init()
  if(!is.null(models)) {
    lapply(models, function(model) {
      cli::cli_h1("Downloading model {.val {model}}")
      cli::cli_alert("(1) Downloading configuration...")
      transformers$AutoConfig$from_pretrained(model)
      cli::cli_alert("(2) Downloading tokenizer...")
      transformers$AutoTokenizer$from_pretrained(model)
      cli::cli_alert("(3) Downloading model...")
      transformers$AutoModel$from_pretrained(model)
      cli::cli_alert_success("Successfully downloaded model {.val {model}}")
      gc()
    })
  }
  cache.folder = str_replace_all(transformers$TRANSFORMERS_CACHE, "\\\\", "/")
  cache.sizegb = sum(file.size(list.files(cache.folder, recursive=TRUE, full.names=TRUE))) / 1024^3
  local.models = find_cached_models(cache.folder)
  cli::cli_h2("Downloaded models:")
  print(local.models)
  cat("\n")
  cli::cli_alert_success("Downloaded models saved at {.path {cache.folder}} ({sprintf('%.2f', cache.sizegb)} GB)")
}


#### FMAT ####


#' (Down)Load BERT models (useless for GPU).
#'
#' Load BERT models from local cache folder "%USERPROFILE%/.cache/huggingface".
#' Models that have not been downloaded can also
#' be automatically downloaded (but *silently*).
#' For [GPU Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration),
#' please directly use [`FMAT_run`] instead.
#'
#' @inheritParams BERT_download
#'
#' @return
#' A named list of fill-mask pipelines obtained from the models.
#' The returned object *cannot* be saved as any RData.
#' You will need to *rerun* this function if you *restart* the R session.
#'
#' @seealso
#' [`BERT_download`]
#'
#' [`FMAT_query`]
#'
#' [`FMAT_query_bind`]
#'
#' [`FMAT_run`]
#'
#' @examples
#' \dontrun{
#' model.names = c("bert-base-uncased", "bert-base-cased")
#' models = FMAT_load(model.names)  # load models from cache
#' }
#'
#' @export
FMAT_load = function(models) {
  transformers = transformers_init()
  cache.folder = str_replace_all(transformers$TRANSFORMERS_CACHE, "\\\\", "/")
  cli::cli_text("Loading models from {.path {cache.folder}} ...")
  fms = lapply(models, function(model) {
    t0 = Sys.time()
    reticulate::py_capture_output({
      fill_mask = transformers$pipeline("fill-mask", model=model)
    })
    cli::cli_alert_success("{model} ({dtime(t0)})")
    return(list(model.name=model, fill.mask=fill_mask))
  })
  names(fms) = models
  class(fms) = "fill.mask"
  return(fms)
}


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
# expand_pair(query, .(High=cc("high, strong"), Low=cc("low, weak")))
# expand_pair(query, .(H="high", M="medium", L="low"))
# X = .(Flower=cc("rose, iris, lily"), Pos=cc("health, happiness, love, peace"))
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
  dq = purrr::map_df(.x, .f, ...) # .x should be query (chr vec)
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
#' @param query Query text (should be a character string/vector
#' with at least one `[MASK]` token).
#' Multiple queries share the same set of
#' `MASK`, `TARGET`, and `ATTRIB`.
#' For multiple queries with different
#' `MASK`, `TARGET`, and/or `ATTRIB`,
#' please use [`FMAT_query_bind`] to combine them.
#' @param MASK A named list of `[MASK]` target words.
#' Must be single words in the vocabulary of a certain masked language model.
#'
#' For model vocabulary, see, e.g.,
#' <https://huggingface.co/bert-base-uncased/raw/main/vocab.txt>
#'
#' Infrequent words may be not included in a model's vocabulary,
#' and in this case you may insert the words into the context by
#' specifying either `TARGET` or `ATTRIB`.
#' @param TARGET,ATTRIB A named list of Target/Attribute words or phrases.
#' If specified, then `query` must contain
#' `{TARGET}` and/or `{ATTRIB}` (in all uppercase and in braces)
#' to be replaced by the words/phrases.
#' @param unmask.id If multiple `[MASK]` are in `query`,
#' it determines which one should be unmasked.
#' Defaults to the 1st `[MASK]`.
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' [`FMAT_load`]
#'
#' [`FMAT_query_bind`]
#'
#' [`FMAT_run`]
#'
#' @examples
#' FMAT_query("[MASK] is a nurse.", MASK = .(Male="He", Female="She"))
#'
#' FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=cc("a doctor, a nurse, an artist"))
#' )
#'
#' FMAT_query(
#'   "The [MASK] {ATTRIB}.",
#'   MASK = .(Male=cc("man, boy"),
#'            Female=cc("woman, girl")),
#'   ATTRIB = .(Masc=cc("is masculine, has a masculine personality"),
#'              Femi=cc("is feminine, has a feminine personality"))
#' )
#'
#' FMAT_query(
#'   "The association between {TARGET} and {ATTRIB} is [MASK].",
#'   MASK = .(H="strong", L="weak"),
#'   TARGET = .(Flower=cc("rose, iris, lily"),
#'              Insect=cc("ant, cockroach, spider")),
#'   ATTRIB = .(Pos=cc("health, happiness, love, peace"),
#'              Neg=cc("death, sickness, hatred, disaster"))
#' )
#'
#' @export
FMAT_query = function(
    query = "Text with [MASK], optionally with {TARGET} and/or {ATTRIB}.",
    MASK = .(),
    TARGET = .(),
    ATTRIB = .(),
    unmask.id = 1
) {
  if(any(str_detect(query, "\\[MASK\\]", negate=TRUE)))
    stop("`query` should contain a [MASK] token!", call.=FALSE)
  if(length(MASK) == 0) {
    stop("Please specify `MASK` (the targets of [MASK])!", call.=FALSE)
  } else {
    MASK = fix_pair(MASK)
  }

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
      rbind(
        expand_full(q, c(target[1], attrib[1])),
        expand_full(q, c(target[1], attrib[2])),
        expand_full(q, c(target[2], attrib[1])),
        expand_full(q, c(target[2], attrib[2]))
      )
    }, TARGET, ATTRIB)
    dq = plyr::adply(dx, 1, function(x) cbind(dm, x))
  }

  if(any(str_count(query, "\\[MASK\\]") > 1))
    dq$unmask.id = as.integer(unmask.id)
  if(length(query) > 1)
    dq = cbind(data.table(qid = as.factor(as.numeric(dq$query))), dq)
  attr(dq, "type") = type
  return(na.omit(dq[order(query)]))
}


#' Combine multiple query data.tables and renumber query ids.
#'
#' @param ... Query data.tables returned from [`FMAT_query`].
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' [`FMAT_load`]
#'
#' [`FMAT_query`]
#'
#' [`FMAT_run`]
#'
#' @examples
#' FMAT_query_bind(
#'   FMAT_query(
#'     "[MASK] is {TARGET}.",
#'     MASK = .(Male="He", Female="She"),
#'     TARGET = .(Occupation=cc("a doctor, a nurse, an artist"))
#'   ),
#'   FMAT_query(
#'     "[MASK] occupation is {TARGET}.",
#'     MASK = .(Male="His", Female="Her"),
#'     TARGET = .(Occupation=cc("doctor, nurse, artist"))
#'   )
#' )
#'
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


# Using plyr!
# library(plyr)
# library(doParallel)
#
# minmax = function(d) data.frame(min=min(d$yield), max=max(d$yield))
#
# adply(npk, 1, minmax, .progress="time")
#
# cl = makeCluster(detectCores())
# registerDoParallel(cl)
# suppressWarnings(adply(npk, 1, minmax, .parallel=TRUE))
# stopCluster(cl)


#' Run the fill-mask pipeline on multiple models (CPU / GPU).
#'
#' Run the fill-mask pipeline on multiple models with CPU or GPU
#' (faster but requiring an NVIDIA GPU device).
#'
#' @details
#' The function automatically adjusts for
#' the compatibility of tokens used in certain models:
#' (1) for uncased models (e.g., ALBERT), it turns tokens to lowercase;
#' (2) for models that use `<mask>` rather than `[MASK]`,
#' it automatically uses the corrected mask token;
#' (3) for models that require a prefix to estimate whole words than subwords
#' (e.g., ALBERT, RoBERTa), it adds a certain prefix (usually a white space;
#' \\u2581 for ALBERT and XLM-RoBERTa, \\u0120 for RoBERTa and DistilRoBERTa).
#'
#' Note that these changes only affect the `token` variable
#' in the returned data, but will not affect the `M_word` variable.
#' Thus, users may analyze data based on the unchanged `M_word`
#' rather than the `token`.
#'
#' Note also that there may be extremely trivial differences
#' (after 5~6 significant digits) in the
#' raw probability estimates between using CPU and GPU,
#' but these differences would have little impact on main results.
#'
#' @param models Options:
#' - A character vector of model names at
#'   [HuggingFace](https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers).
#'   - Can be used for both CPU and GPU.
#' - A returned object from [`FMAT_load`].
#'   - Can ONLY be used for CPU.
#'   - If you *restart* the R session,
#'     you will need to *rerun* [`FMAT_load`].
#' @param data A data.table returned from [`FMAT_query`] or [`FMAT_query_bind`].
#' @param gpu Use GPU (3x faster than CPU) to run the fill-mask pipeline?
#' Defaults to missing value that will *automatically* use available GPU
#' (if not available, then use CPU).
#' An NVIDIA GPU device (e.g., GeForce RTX Series) is required to use GPU.
#' See [Guidance for GPU Acceleration](https://psychbruce.github.io/FMAT/#guidance-for-gpu-acceleration).
#'
#' Options passing to the `device` parameter in Python:
#' - `FALSE`: CPU (`device = -1`).
#' - `TRUE`: GPU (`device = 0`).
#' - Any other value: passing to
#'   [transformers.pipeline(device=...)](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.device)
#'   which defines the device (e.g.,
#'   `"cpu"`, `"cuda:0"`, or a GPU device id like `1`)
#'   on which the pipeline will be allocated.
#' @param file File name of `.RData` to save the returned data.
#' @param progress Show a progress bar? Defaults to `TRUE`.
#' @param warning Alert warning of out-of-vocabulary word(s)? Defaults to `TRUE`.
#'
#' @return
#' A data.table (of new class `fmat`) appending `data` with these new variables:
#' - `model`: model name.
#' - `output`: complete sentence output with unmasked token.
#' - `token`: actual token to be filled in the blank mask
#'   (a note "out-of-vocabulary" will be added
#'   if the original word is not found in the model vocabulary).
#' - `prob`: (raw) conditional probability of the unmasked token
#'   given the provided context, estimated by the masked language model.
#'   - It is NOT SUGGESTED to directly interpret the raw probabilities
#'   because the *contrast* between a pair of probabilities
#'   is more interpretable. See [`summary.fmat`].
#'
#' @seealso
#' [`BERT_download`]
#'
#' [`FMAT_load`]
#'
#' [`FMAT_query`]
#'
#' [`FMAT_query_bind`]
#'
#' [`summary.fmat`]
#'
#' @examples
#' ## Running the examples requires the models downloaded
#'
#' \dontrun{
#' models = FMAT_load(c("bert-base-uncased", "bert-base-cased"))
#' # for GPU acceleration, please use `FMAT_run()` directly
#'
#' query1 = FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=cc("a doctor, a nurse, an artist"))
#' )
#' data1 = FMAT_run(models, query1)
#' summary(data1, target.pair=FALSE)
#'
#' query2 = FMAT_query(
#'   "The [MASK] {ATTRIB}.",
#'   MASK = .(Male=cc("man, boy"),
#'            Female=cc("woman, girl")),
#'   ATTRIB = .(Masc=cc("is masculine, has a masculine personality"),
#'              Femi=cc("is feminine, has a feminine personality"))
#' )
#' data2 = FMAT_run(models, query2)
#' summary(data2, mask.pair=FALSE)
#' summary(data2)
#'
#' query3 = FMAT_query(
#'   "The association between {TARGET} and {ATTRIB} is [MASK].",
#'   MASK = .(H="strong", L="weak"),
#'   TARGET = .(Flower=cc("rose, iris, lily"),
#'              Insect=cc("ant, cockroach, spider")),
#'   ATTRIB = .(Pos=cc("health, happiness, love, peace"),
#'              Neg=cc("death, sickness, hatred, disaster"))
#' )
#' data3 = FMAT_run(models, query3)
#' summary(data3, attrib.pair=FALSE)
#' summary(data3)
#' }
#'
#' @export
FMAT_run = function(
    models,
    data,
    gpu,
    file = NULL,
    progress = TRUE,
    warning = TRUE
) {
  t0 = Sys.time()
  type = attr(data, "type")
  device = gpu_to_device(gpu)
  progress = ifelse(progress, "text", "none")

  if(inherits(models, "fill.mask")) {
    if(!device %in% c(-1L, "cpu"))
      stop("
      To use GPU, please specify `models` as model names,
      rather than the returned object from `FMAT_load()`.", call.=FALSE)
  } else {
    transformers = transformers_init()
    cache.folder = str_replace_all(transformers$TRANSFORMERS_CACHE, "\\\\", "/")
    cli::cli_text("Loading models from {.path {cache.folder}} ...")
    cat("\n")
  }

  onerun = function(model, data=data) {
    ## ---- One Run Begin ---- ##
    if(is.character(model)) {
      reticulate::py_capture_output({
        fill_mask = transformers$pipeline("fill-mask", model=model, device=device)
      })
    } else {
      fill_mask = model$fill.mask
      model = model$model.name
    }
    if(device %in% c(-1L, "cpu"))
      cli::cli_h1("{.val {model}} (CPU)")
    else
      cli::cli_h1("{.val {model}} (GPU Accelerated)")

    uncased = str_detect(model, "uncased|albert")
    prefix.u2581 = str_detect(model, "xlm-roberta|albert")
    prefix.u0120 = str_detect(model, "roberta|bertweet-large") & !str_detect(model, "xlm")
    mask.lower = str_detect(model, "roberta|bertweet")

    unmask = function(d) {
      ## unmask function begin ##
      if("TARGET" %in% names(d))
        TARGET = as.character(d$T_word)
      if("ATTRIB" %in% names(d))
        ATTRIB = as.character(d$A_word)
      uid = if("unmask.id" %in% names(d)) d$unmask.id else 1
      query = str_replace_all(d$query, "\\[mask\\]", "[MASK]")
      query = glue::glue(as.character(query))
      mask = as.character(d$M_word)
      mask.begin = str_sub(query, 1, 6) == "[MASK]" & uid == 1
      if(uncased) mask = tolower(mask)
      if(prefix.u2581) mask = paste0("\u2581", mask)
      if(prefix.u0120 & !mask.begin) mask = paste0("\u0120", mask)
      if(mask.lower) query = str_replace_all(query, "\\[MASK\\]", "<mask>")
      out = reticulate::py_capture_output({
        res = fill_mask(query, targets=mask, top_k=1L)[[uid]]
      })
      # UserWarning: You seem to be using the pipelines sequentially on GPU.
      # In order to maximize efficiency please use a dataset.
      return(data.table(
        output = res$sequence,
        token = ifelse(
          out=="" | str_detect(out, "UserWarning|GPU"),  # no extra output from python
          res$token_str,
          paste(res$token_str,
                ifelse(str_detect(out, "vocabulary"),
                       "(out-of-vocabulary)", out))),
        prob = res$score
      ))
      ## unmask function end ##
    }

    t1 = Sys.time()
    suppressWarnings({
      data = plyr::adply(data, 1, unmask, .progress=progress)
    })
    speed = sprintf("%.0f", nrow(data) / as.numeric(difftime(Sys.time(), t1, units="mins")))
    cat(paste0("  (", dtime(t1), ") [", speed, " queries/min]\n"))

    rm(fill_mask)
    gc()

    return(cbind(data.table(model=as.factor(model)), data))
    ## ---- One Run End ---- ##
  }

  cli::cli_alert_info("Task: {length(models)} models * {nrow(data)} queries")
  t0.task = Sys.time()
  data = rbindlist(lapply(models, onerun, data=data))
  speed = sprintf("%.0f", nrow(data) / as.numeric(difftime(Sys.time(), t0.task, units="mins")))
  cat("\n")
  attr(data, "type") = type
  class(data) = c("fmat", class(data))
  gc()
  cli::cli_alert_success("Task completed ({dtime(t0)}) [{speed} queries/min]")

  if(warning) warning_oov(data)

  if(!is.null(file)) {
    if(!str_detect(file, "\\.[Rr][Dd]a(ta)?$"))
      file = paste0(file, ".RData")
    save(data, file=file)
    cli::cli_alert_success("Data saved to {.val {file}}")
  }

  return(data)
}


warning_oov = function(data) {
  d.oov = unique(data[str_detect(data$token, "out-of-vocabulary"),
                      c("M_word", "token")])
  d.oov$token = str_remove(d.oov$token, " \\(out-of-vocabulary\\)")
  if(nrow(d.oov) > 0) {
    oov0 = unique(d.oov$M_word)
    for(oov in oov0) {
      di = d.oov[d.oov$M_word==oov]
      cli::cli_alert_warning("
      Replaced out-of-vocabulary word {.val {oov}} by: {.val {di$token}}")
    }
  }
}


#' \[S3 method\] Summarize the results for the FMAT.
#'
#' @description
#' Summarize the results of *Log Probability Ratio* (LPR),
#' which indicates the *relative* (vs. *absolute*)
#' association between concepts.
#'
#' The LPR of just one contrast (e.g., only between a pair of attributes)
#' may *not* be sufficient for a proper interpretation of the results,
#' and may further require a second contrast (e.g., between a pair of targets).
#'
#' Users are suggested to use linear mixed models
#' (with the R packages `nlme` or `lme4`/`lmerTest`)
#' to perform the formal analyses and hypothesis tests based on the LPR.
#'
#' @inheritParams FMAT_run
#' @param object A data.table (of new class `fmat`)
#' returned from [`FMAT_run`].
#' @param mask.pair,target.pair,attrib.pair Pairwise contrast of
#' `[MASK]`, `TARGET`, `ATTRIB`?
#' Defaults to `TRUE`.
#' @param ... Other arguments (currently not used).
#'
#' @return
#' A data.table of the summarized results with Log Probability Ratio (LPR).
#'
#' @seealso
#' [`FMAT_run`]
#'
#' @examples
#' # see examples in `FMAT_run`
#'
#' @export
summary.fmat = function(
    object,
    mask.pair=TRUE,
    target.pair=TRUE,
    attrib.pair=TRUE,
    warning=TRUE,
    ...) {
  if(warning) warning_oov(object)
  type = attr(object, "type")
  M_word = T_word = A_word = MASK = TARGET = ATTRIB = prob = LPR = NULL

  if(mask.pair) {
    gvars = c("model", "query", "M_pair",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word")
    grouping.vars = intersect(names(object), gvars)
    dt = object[, .(
      MASK = paste(MASK[1], "-", MASK[2]),
      M_word = paste(M_word[1], "-", M_word[2]),
      LPR = log(prob[1]) - log(prob[2])
    ), keyby = grouping.vars]
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
        TARGET = paste(TARGET[1], "-", TARGET[2]),
        T_word = paste(T_word[1], "-", T_word[2]),
        LPR = LPR[1] - LPR[2]
      ), keyby = c("model", "query", "MASK", "M_word", "T_pair")]
      dt$TARGET = as_factor(dt$TARGET)
      dt$T_word = as_factor(dt$T_word)
      dt$T_pair = NULL
    }
  }

  if(type=="MA") {
    if(attrib.pair) {
      dt = dt[, .(
        ATTRIB = paste(ATTRIB[1], "-", ATTRIB[2]),
        A_word = paste(A_word[1], "-", A_word[2]),
        LPR = LPR[1] - LPR[2]
      ), keyby = c("model", "query", "MASK", "M_word", "A_pair")]
      dt$ATTRIB = as_factor(dt$ATTRIB)
      dt$A_word = as_factor(dt$A_word)
      dt$A_pair = NULL
    }
  }

  if(type=="MTA") {
    if(attrib.pair) {
      dt = dt[, .(
        LPR = mean(LPR)
      ), keyby = c("model", "query", "MASK", "M_word",
                   "TARGET", "T_word", "ATTRIB")]
      dt = dt[, .(
        ATTRIB = paste(ATTRIB[1], "-", ATTRIB[2]),
        LPR = LPR[1] - LPR[2]
      ), keyby = c("model", "query", "MASK", "M_word",
                   "TARGET", "T_word")]
    }
  }

  return(dt)
}


#' Reliability analysis (Cronbach's \eqn{\alpha}) of LPR.
#'
#' @param fmat A data.table returned from [`summary.fmat`].
#' @param item Reliability of multiple `"query"` (default),
#' `"T_word"`, or `"A_word"`.
#' @param by Variable(s) to split data by.
#' Options can be `"model"`, `"TARGET"`, `"ATTRIB"`,
#' or any combination of them.
#'
#' @return A data.table of Cronbach's \eqn{\alpha}.
#'
#' @export
LPR_reliability = function(
    fmat,
    item=c("query", "T_word", "A_word"),
    by=NULL) {
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
        alpha = psych::alpha(dplyr::select(x, tidyr::starts_with("LPR")),
                             delete=FALSE, warnings=FALSE)
      })
    })
    data.frame(n.obs = nrow(x),
               k.items = alpha$nvar,
               alpha = alpha$total$raw_alpha)
  })
  if(is.null(by)) alphas[[1]] = NULL
  return(as.data.table(alphas))
}


#### Deprecated ####


## Install Python modules and initialize local environment.
##
## Install required Python modules and
## initialize a local "conda" environment.
## Run this function only once after you have installed the package.
##
## @examples
## \dontrun{
## FMAT_init()  # run it only once
##
## # Then please specify the version of Python:
## # RStudio -> Tools -> Global/Project Options
## # -> Python -> Select -> Conda Environments
## # -> Choose ".../textrpp_condaenv/python.exe"
## }
##
## @export
# FMAT_init = function() {
#   suppressMessages({
#     suppressWarnings({
#       text::textrpp_install(prompt=FALSE)
#     })
#   })
#   cat("\n")
#   cli::cli_alert_success("{.pkg Successfully installed Python modules in conda environment.}")
#
#   try({
#     error = TRUE
#     suppressMessages({
#       suppressWarnings({
#         text::textrpp_initialize(save_profile=TRUE, prompt=FALSE)
#       })
#     })
#     error = FALSE
#   }, silent=TRUE)
#   if(error)
#     stop("
#
#       Please specify the version of Python:
#         RStudio -> Tools -> Global/Project Options
#         -> Python -> Select -> Conda Environments
#         -> Choose \".../textrpp_condaenv/python.exe\"",
#        call.=FALSE)
#   cat("\n")
#   cli::cli_alert_success("{.pkg Initialized the Python modules.}")
# }


# if(parallel) {
#   cl = parallel::makeCluster(ncores)
#   models = names(models)
#   data = rbindlist(parallel::parLapply(cl, models, onerun, data=data))
#   parallel::stopCluster(cl)
# } else {
#   data = rbindlist(lapply(models, onerun, data=data))
#   cat("\n")
# }

