#### Initialize ####


#' @import stringr
#' @import data.table
#' @importFrom forcats as_factor
#' @importFrom stats na.omit
.onAttach = function(libname, pkgname) {
  inst.ver = as.character(utils::packageVersion("FMAT"))
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

    "))
  }
}


#### Basic ####


#' @export
. = list


#' @importFrom PsychWordVec cc
#' @export
PsychWordVec::cc


text_initialized = function() {
  error = TRUE
  try({
    text::textModels()
    error = FALSE
  }, silent=TRUE)
  if(error) PsychWordVec::text_init()
}


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


#### FMAT ####


#' Initialize running environment and (down)load language models.
#'
#' @param models Language model names (usually the BERT-based models) at
#' \href{https://huggingface.co/models}{HuggingFace}.
#'
#' For a full list of available models, see
#' \url{https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers}
#'
#' @return
#' A named list of fill-mask pipelines obtained from the models.
#' The returned object \emph{cannot} be saved as any RData.
#' You will need to \emph{rerun} this function if you restart the R session.
#'
#' @seealso
#' \code{\link{FMAT_query}}
#'
#' \code{\link{FMAT_query_bind}}
#'
#' \code{\link{FMAT_run}}
#'
#' @examples
#' \donttest{models = FMAT_load(c("bert-base-uncased", "bert-base-cased"))}
#'
#' @export
FMAT_load = function(models) {
  cli::cli_text("Initializing environment...")
  text_initialized()
  old.models = text::textModels()$Downloaded_models
  new.models = setdiff(models, old.models)
  if(length(new.models) > 0) PsychWordVec::text_model_download(new.models)

  cli::cli_text("Loading models...")
  transformers = reticulate::import("transformers")
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
#' with at least one \code{[MASK]} token).
#' Multiple queries share the same set of
#' \code{MASK}, \code{TARGET}, and \code{ATTRIB}.
#' For multiple queries with different
#' \code{MASK}, \code{TARGET}, and/or \code{ATTRIB},
#' please use \code{\link{FMAT_query_bind}} to combine them.
#' @param MASK A named list of \code{[MASK]} target words.
#' Must be single words in the vocabulary of a certain masked language model.
#' For model vocabulary, see, e.g.,
#' \url{https://huggingface.co/bert-base-uncased/raw/main/vocab.txt}
#'
#' Note that infrequent words may be not included in a model's vocabulary,
#' and in this case you may insert the words into the context by
#' specifying either \code{TARGET} or \code{ATTRIB}.
#' @param TARGET,ATTRIB A named list of Target/Attribute words or phrases.
#' If specified, then \code{query} must contain
#' \code{{TARGET}} and/or \code{{ATTRIB}} (in all uppercase and in braces)
#' to be replaced by the words/phrases.
#' @param unmask.id If there are multiple \code{[MASK]} in \code{query},
#' this argument will be used to determine which one is to be unmasked.
#' Defaults to the 1st \code{[MASK]}.
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' \code{\link{FMAT_load}}
#'
#' \code{\link{FMAT_query_bind}}
#'
#' \code{\link{FMAT_run}}
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
#'   "The {TARGET} has a [MASK] association with {ATTRIB}.",
#'   MASK = .(H="high", L="low"),
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
#' @param ... Query data.tables returned from \code{\link{FMAT_query}}.
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' \code{\link{FMAT_load}}
#'
#' \code{\link{FMAT_query}}
#'
#' \code{\link{FMAT_run}}
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


#' Run the mask filling pipeline on multiple models.
#'
#' @details
#' The function will also automatically adjust for
#' the compatibility of tokens used in certain models:
#' (1) for uncased models (e.g., ALBERT), it turns tokens to lowercase;
#' (2) for models that use \code{<mask>} rather than \code{[MASK]},
#' it automatically uses the corrected mask token;
#' (3) for models that require a prefix to estimate whole words than subwords
#' (e.g., ALBERT, RoBERTa), it adds a certain prefix (usually a white space;
#' \\u2581 for ALBERT and XLM-RoBERTa, \\u0120 for RoBERTa and DistilRoBERTa).
#'
#' Note that these changes only affect the \code{token} variable
#' in the returned data, but will not affect the \code{M_word} variable.z
#' Thus, users may analyze their data based on the unchanged \code{M_word}
#' rather than the \code{token}.
#'
#' @param models Language model(s):
#' \itemize{
#'   \item{Model names (usually the BERT-based models) at
#'    \href{https://huggingface.co/models}{HuggingFace}.}
#'   \item{A list of mask filling pipelines loaded by \code{\link{FMAT_load}}.
#'
#'    * You will need to \strong{rerun} \code{\link{FMAT_load}}
#'    if you \strong{restart} the R session.}
#' }
#' @param data A data.table returned from
#' \code{\link{FMAT_query}} or \code{\link{FMAT_query_bind}}.
#' @param file File name of \code{.RData} to save the returned data.
#' @param progress Show a progress bar:
#' \code{"none"} (\code{FALSE}), \code{"text"} (\code{TRUE}), \code{"time"}.
#' @param parallel Parallel processing. Defaults to \code{FALSE}.
#' If \code{TRUE}, then \code{models} must be model names
#' rather than from \code{\link{FMAT_load}}.
#'
#' * For small-scale \code{data},
#' parallel processing would instead be \emph{slower}
#' because it takes time to create a parallel cluster.
#' @param ncores Number of CPU cores to be used in parallel processing.
#' Defaults to the minimum of the number of models and your CPU cores.
#' @param warning Warning of out-of-vocabulary word(s). Defaults to \code{TRUE}.
#'
#' @return
#' A data.table (of new class \code{fmat}) appending \code{data}
#' with these new variables:
#' \itemize{
#'   \item{\code{model}: model name.}
#'   \item{\code{output}: complete sentence output with unmasked token.}
#'   \item{\code{token}: actual token to be filled in the blank mask
#'   (a note "out-of-vocabulary" will be added
#'   if the original word is not found in the model vocabulary).}
#'   \item{\code{prop}: (raw) conditional probability of the unmasked token
#'   given the provided context, estimated by the masked language model.
#'
#'   * It is NOT SUGGESTED to directly interpret the raw probabilities
#'   because the \emph{contrast} between a pair of probabilities
#'   is more interpretable. See \code{\link{summary.fmat}}.}
#' }
#'
#' @seealso
#' \code{\link{FMAT_load}}
#'
#' \code{\link{FMAT_query}}
#'
#' \code{\link{FMAT_query_bind}}
#'
#' \code{\link{summary.fmat}}
#'
#' @examples
#' # Running the example requires the models downloaded
#' # You will need to rerun `FMAT_load` if you restart the R session
#' \donttest{
#' models = FMAT_load(c("bert-base-uncased", "bert-base-cased"))
#'
#' dq = FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=cc("a doctor, a nurse, an artist"))
#' )
#' data1 = FMAT_run(models, dq)
#' summary(data1)
#'
#' data2 = FMAT_run(
#'   models,
#'   FMAT_query(
#'     "The {TARGET} has a [MASK] association with {ATTRIB}.",
#'     MASK = .(H="high", L="low"),
#'     TARGET = .(Flower=cc("rose, iris, lily"),
#'                Insect=cc("ant, cockroach, spider")),
#'     ATTRIB = .(Pos=cc("health, happiness, love, peace"),
#'                Neg=cc("death, sickness, hatred, disaster"))
#'   ))
#' summary(data2)
#' }
#' @export
FMAT_run = function(
    models,
    data,
    file = NULL,
    progress = c(FALSE, TRUE, "none", "text", "time"),
    parallel = FALSE,
    ncores = min(length(models), parallel::detectCores()),
    warning = TRUE
) {
  t0 = Sys.time()
  progress = match.arg(progress)
  if(progress=="FALSE") progress = "none"
  if(progress=="TRUE") progress = "text"
  type = attr(data, "type")

  text_initialized()
  cli::cli_alert_success("Environment initialized ({dtime(t0)})")

  onerun = function(model, data=data) {
    if(is.character(model)) {
      t1 = Sys.time()
      transformers = reticulate::import("transformers")
      reticulate::py_capture_output({
        fill_mask = transformers$pipeline(task="fill-mask", model=model)
      })
      cli::cli_h1("{model} (model loaded: {dtime(t1)})")
    }
    if(is.list(model)) {
      fill_mask = model$fill.mask
      model = model$model.name
      cli::cli_h1("{model}")
    }

    uncased = str_detect(model, "uncased|albert")
    prefix.u2581 = str_detect(model, "xlm-roberta|albert")
    prefix.u0120 = str_detect(model, "roberta") & !str_detect(model, "xlm")
    mask.lower = str_detect(model, "roberta")

    unmask = function(d) {
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
      oov = reticulate::py_capture_output({
        res = fill_mask(query, targets=mask, top_k=1L)[[uid]]
      })
      return(data.table(
        output = res$sequence,
        token = ifelse(
          oov=="",  # no extra output from python
          res$token_str,
          paste(res$token_str, "(out-of-vocabulary)")),
        prop = res$score
      ))
    }

    t2 = Sys.time()
    suppressWarnings({
      data = plyr::adply(
        data, 1, unmask,
        .progress = if(parallel) "none" else progress
      )
    })
    cat(paste0("  (", dtime(t2), ")\n"))

    return(cbind(data.table(model=as.factor(model)), data))
  }

  cli::cli_alert_info(" Task: {length(models)} models * {nrow(data)} queries")

  if(parallel) {
    cl = parallel::makeCluster(ncores)
    models = names(models)
    data = rbindlist(parallel::parLapply(cl, models, onerun, data=data))
    parallel::stopCluster(cl)
  } else {
    data = rbindlist(lapply(models, onerun, data=data))
    cat("\n")
  }
  attr(data, "type") = type
  class(data) = c("fmat", class(data))

  gc()
  cli::cli_alert_success("Task completed (total time cost = {dtime(t0)})")

  if(warning) warning_oov(data)

  if(!is.null(file)) {
    if(!str_detect(file, "\\.[Rr][Dd]a|\\.[Rr][Dd]ata"))
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


#' [S3 method] Summarize the results for the FMAT.
#'
#' @description
#' Summarize the results of \emph{Log Probability Ratio} (LPR),
#' which indicates the \emph{relative} (vs. \emph{absolute})
#' association between concepts.
#'
#' The LPR of just one contrast (e.g., only between a pair of attributes)
#' may \emph{not} be sufficient for a proper interpretation of the results,
#' and may further require a second contrast (e.g., between a pair of targets).
#'
#' Users are encouraged to use linear mixed models
#' (with the R packages \code{nlme} or \code{lme4}/\code{lmerTest})
#' to perform the formal analyses and hypothesis tests based on the LPR.
#'
#' @inheritParams FMAT_run
#' @param object A data.table (of new class \code{fmat})
#' returned from \code{\link{FMAT_run}}.
## @param digits Number of decimal places of output. Defaults to \code{3}.
#' @param mask.pair,target.pair,attrib.pair Pairwise contrast of
#' \code{[MASK]}, \code{TARGET}, \code{ATTRIB}?
#' Defaults to \code{TRUE}.
#' @param ... Other arguments (currently not used).
#'
#' @return
#' A data.table of the summarized results with Log Probability Ratio (LPR).
#'
#' @seealso
#' \code{\link{FMAT_run}}
#'
#' @export
summary.fmat = function(object,
                        mask.pair=TRUE,
                        target.pair=TRUE,
                        attrib.pair=TRUE,
                        warning=TRUE,
                        ...) {
  if(warning) warning_oov(object)
  type = attr(object, "type")
  M_word = T_word = A_word = MASK = TARGET = ATTRIB = prop = LPR = NULL

  if(mask.pair) {
    gvars = c("model", "query", "M_pair",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word")
    grouping.vars = intersect(names(object), gvars)
    dt = object[, .(
      MASK = paste(MASK[1], "-", MASK[2]),
      M_word = paste(M_word[1], "-", M_word[2]),
      LPR = log(prop[1]) - log(prop[2])
    ), keyby = grouping.vars]
    dt$MASK = as_factor(dt$MASK)
    dt$M_word = as_factor(dt$M_word)
    dt$M_pair = NULL
  } else {
    dvars = c("model", "query", "MASK", "M_word",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word",
              "prop")
    dt.vars = intersect(names(object), dvars)
    dt = object[, dt.vars, with=FALSE]
    dt$LPR = log(dt$prop)
    dt$prop = NULL
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

  return(dt)
}


