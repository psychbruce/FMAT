#### Initialize ####


#' @import stringr
#' @import data.table
#' @importFrom forcats as_factor
.onAttach = function(libname, pkgname) {
  inst.ver = as.character(utils::packageVersion("FMAT"))
  pkgs = c("data.table", "stringr")
  suppressMessages({
    suppressWarnings({
      loaded = sapply(pkgs, require, character.only=TRUE)
    })
  })
  if(all(loaded)) {
    cli::cli_h1("FMAT (v{inst.ver})")
    cn()
    cli::cli_alert_success("
    Packages also loaded: {.pkg data.table, stringr}
    ")
    cn()
  }
}


#### Basic ####


cn = function() cat("\n")

#' A wrapper of \code{list()}.
#'
#' A simple version of the \code{\link{list}} function.
#'
#' @param ... Objects (possibly named) passed to \code{\link{list}}.
#'
#' @return A list.
#'
#' @examples
#' .(Male=s("man, he, his"), Female=s("woman, she, her"))
#'
#' @export
. = function(...) list(...)

#' Split a string (with separators) into a character vector.
#'
#' @param x Character string.
#' Separators can be: \code{,} \code{;} \code{|} \code{\\n} \code{\\t}.
#'
#' @return Character vector.
#'
#' @examples
#' s("a, b, c, d, e")
#'
#' @export
s = function(x) {
  as.character(str_split(str_trim(x), "\\s*[,;\\|\\n\\t]\\s*", simplify=TRUE))
}

text_init = function() {
  suppressMessages({
    suppressWarnings({
      text::textrpp_install(prompt=FALSE)
    })
  })
  cn()
  cli::cli_alert_success("{.pkg Installed Python modules in conda environment.}")

  error = TRUE
  try({
    suppressMessages({
      suppressWarnings({
        text::textrpp_initialize(save_profile=TRUE, prompt=FALSE)
      })
    })
    error = FALSE
  }, silent=TRUE)
  if(error)
    stop("No valid Python or conda environment.

       You may need to specify the version of Python:
         RStudio -> Tools -> Global/Project Options
         -> Python -> Select -> Conda Environments
         -> Choose \".../textrpp_condaenv/python.exe\"",
       call.=FALSE)
  cn()
  cli::cli_alert_success("{.pkg Initialized the Python modules.}")
}

text_initialized = function() {
  error = TRUE
  try({
    text::textModels()
    error = FALSE
  }, silent=TRUE)
  if(error) text_init()
}

model_download = function(model=NULL) {
  text_initialized()
  if(!is.null(model)) {
    for(m in model) {
      cli::cli_h1("Downloading model \"{m}\"")
      transformers = reticulate::import("transformers")
      cli::cli_text("Downloading configuration...")
      config = transformers$AutoConfig$from_pretrained(m)
      cli::cli_text("Downloading tokenizer...")
      tokenizer = transformers$AutoTokenizer$from_pretrained(m)
      cli::cli_text("Downloading model...")
      model = transformers$AutoModel$from_pretrained(m)
      cli::cli_alert_success("Successfully downloaded model \"{m}\"")
      gc()
    }
  }
  cli::cli_h2("Currently downloaded language models:")
  models = text::textModels()
  models[[1]] = sort(models[[1]])
  models[[2]] = sort(models[[2]])
  cli::cli_li(paste0("\"", models$Downloaded_models, "\""))
  cn()
  invisible(models)
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
#' \href{
#' https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers
#' }{HuggingFace}.
#'
#' @return
#' A named list of fill-mask pipelines obtained from the models.
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
  if(length(new.models) > 0) model_download(new.models)

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


# query = "[MASK] is ABC."
# expand_pair(query, .(High=s("high, strong"), Low=s("low, weak")))
# expand_pair(query, .(H="high", M="medium", L="low"))
# X = .(Flower=s("rose, iris, lily"), Pos=s("health, happiness, love, peace"))
# expand_full(query, X)

expand_pair = function(query, X, var="MASK") {
  d = data.frame(X)
  dt = rbindlist(lapply(names(d), function(var) {
    data.table(query=query, var=var, pair=seq_len(nrow(d)), word=d[[var]])
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

  dx = data.frame(X)
  dx = rbindlist(lapply(names(dx), function(x) {
    data.table(x=rep(x, each=n),
               pair=rep(seq_len(nrow(dx)), each=n),
               word=rep(dx[[x]], each=n))
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
#' Multiple queries would share the same
#' \code{MASK}, \code{TARGET}, and \code{ATTRIB}.
#' For multiple queries with different
#' \code{MASK}, \code{TARGET}, and/or \code{ATTRIB},
#' please use \code{\link{FMAT_query_bind}} to combine them.
#' @param MASK A named list of \code{[MASK]} target words.
#' Must be single words in the vocabulary of a certain masked language model.
#' Note that infrequent words usually do not exist in a model's vocabulary,
#' and in such a situation you may insert the words into the context by
#' specifying either \code{TARGET} or \code{ATTRIB}.
#' @param TARGET,ATTRIB A named list of Target/Attribute words or phrases.
#' If specified, then \code{query} must contain
#' \code{{TARGET}} and/or \code{{ATTRIB}} that would be
#' replaced by the words/phrases.
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
#'   TARGET = .(Occupation=s("a doctor, a nurse, an artist"))
#' )
#'
#' FMAT_query(
#'   "The [MASK] {ATTRIB}.",
#'   MASK = .(Male=s("man, boy"), Female=s("woman, girl")),
#'   ATTRIB = .(Masc=s("is masculine, has a masculine personality"),
#'              Femi=s("is feminine, has a feminine personality"))
#' )
#'
#' FMAT_query(
#'   "The {TARGET} has a [MASK] association with {ATTRIB}.",
#'   MASK = .(H="high", L="low"),
#'   TARGET = .(Flower=s("rose, iris, lily"),
#'              Insect=s("ant, cockroach, spider")),
#'   ATTRIB = .(Pos=s("health, happiness, love, peace"),
#'              Neg=s("death, sickness, hatred, disaster"))
#' )
#'
#' @export
FMAT_query = function(
    query = "[MASK] must be in query, {TARGET} and {ATTRIB} are optional.",
    MASK = .(),
    TARGET = .(),
    ATTRIB = .(),
    unmask.id = 1
) {
  if(any(str_detect(query, "\\[MASK\\]", negate=TRUE)))
    stop("`query` should contain a [MASK] token!", call.=FALSE)
  if(length(MASK)==0) {
    stop("Please specify `MASK` (the targets of [MASK])!", call.=FALSE)
  } else if(length(TARGET)==0 & length(ATTRIB)==0) {
    # No TARGET or ATTRIB
    type = "M"
    dq = map_query(query, expand_pair, MASK)
  } else if(length(TARGET)>0 & length(ATTRIB)==0) {
    # Only TARGET
    type = "MT"
    dq = append_X(map_query(query, expand_pair, MASK),
                  TARGET, "TARGET")
  } else if(length(TARGET)==0 & length(ATTRIB)>0) {
    # Only ATTRIB
    type = "MA"
    dq = append_X(map_query(query, expand_pair, MASK),
                  ATTRIB, "ATTRIB")
  } else if(length(TARGET)>0 & length(ATTRIB)>0) {
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
  return(dq[order(query)])
}


#' Combine multiple query data.tables and renumber query ids.
#'
#' @param ... Query data.tables returned by \code{\link{FMAT_query}}.
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
#'     TARGET = .(Occupation=s("a doctor, a nurse, an artist"))
#'   ),
#'   FMAT_query(
#'     "[MASK] occupation is {TARGET}.",
#'     MASK = .(Male="His", Female="Her"),
#'     TARGET = .(Occupation=s("doctor, nurse, artist"))
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
#' @param models Language model(s):
#' \itemize{
#'   \item{Model names (usually the BERT-based models) at
#'    \href{
#'    https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers
#'    }{HuggingFace}.}
#'   \item{A list of mask filling pipelines loaded by \code{\link{FMAT_load}}.
#'    You should \strong{rerun} \code{\link{FMAT_load}}
#'    if you \strong{restart} the R session.}
#' }
#' @param data A data.table returned by
#' \code{\link{FMAT_query}} or \code{\link{FMAT_query_bind}}.
#' @param progress Show a progress bar:
#' \code{"text"} (default), \code{"time"}, \code{"none"}.
#' @param parallel Parallel processing. Defaults to \code{FALSE}.
#' If \code{TRUE}, then \code{models} must be model names
#' rather than from \code{\link{FMAT_load}}.
#'
#' Note that for a small \code{data},
#' parallel processing would instead be slower
#' because it takes time to create a parallel cluster.
#' @param ncores Number of CPU cores to be used in parallel processing.
#' Defaults to the minimum of the number of models and your CPU cores.
#'
#' @return
#' A data.table (of new class \code{fmat}) appending \code{data}
#' with these new variables:
#' \itemize{
#'   \item{\code{model}: model name}
#'   \item{\code{output}: complete sentence output with unmasked token}
#'   \item{\code{token}: actual token to be filled in the blank mask
#'   (a note "out-of-vocabulary" will be added
#'   if the original word is not found in the model vocabulary)}
#'   \item{\code{prop}: (raw) conditional probability of the unmasked token
#'   given the context, estimated by the corresponding language model
#'
#'   NOT SUGGESTED to directly interpret the raw probabilities
#'   because the contrast between a pair of probabilities is more meaningful.
#'   See \code{\link{summary.fmat}} for detail.)}
#' }
#'
#' @seealso
#' \code{\link{FMAT_load}}
#'
#' \code{\link{FMAT_query}}
#'
#' \code{\link{FMAT_query_bind}}
#'
#' @examples
#' # Running the example requires models downloaded
#' \donttest{
#' models = FMAT_load(c("bert-base-uncased", "bert-base-cased"))
#'
#' dq = FMAT_query(
#'   c("[MASK] is {TARGET}.", "[MASK] works as {TARGET}."),
#'   MASK = .(Male="He", Female="She"),
#'   TARGET = .(Occupation=s("a doctor, a nurse, an artist"))
#' )
#' data1 = FMAT_run(models, dq)
#' summary(data1)
#'
#' data2 = FMAT_run(
#'   models,
#'   FMAT_query(
#'     "The {TARGET} has a [MASK] association with {ATTRIB}.",
#'     MASK = .(H="high", L="low"),
#'     TARGET = .(Flower=s("rose, iris, lily"),
#'                Insect=s("ant, cockroach, spider")),
#'     ATTRIB = .(Pos=s("health, happiness, love, peace"),
#'                Neg=s("death, sickness, hatred, disaster"))
#'   ))
#' summary(data2)
#' }
#' @export
FMAT_run = function(
    models,
    data,
    progress = c("text", "time", "none"),
    parallel = FALSE,
    ncores = min(length(models), parallel::detectCores())
) {
  t0 = Sys.time()
  progress = match.arg(progress)
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
    prefix = str_detect(model, "xlm-roberta|albert")

    unmask = function(d) {
      if("TARGET" %in% names(d))
        TARGET = as.character(d$T_word)
      if("ATTRIB" %in% names(d))
        ATTRIB = as.character(d$A_word)
      uid = if("unmask.id" %in% names(d)) d$unmask.id else 1
      query = glue::glue(as.character(d$query))
      mask = as.character(d$M_word)
      if(uncased) mask = tolower(mask)
      if(prefix) mask = paste0("\u2581", mask)
      oov = reticulate::py_capture_output({
        res = fill_mask(query, targets=mask, top_k=1L)[[uid]]
      })
      return(data.table(
        output = res$sequence,
        token = ifelse(
          oov=="",
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
    cn()
  }
  attr(data, "type") = type
  class(data) = c("fmat", class(data))

  gc()
  cli::cli_alert_success("Task completed (total time cost = {dtime(t0)})")

  return(data)
}


#' [S3 method] Summarize results of the FMAT.
#'
#' Summarize the results of \emph{Log Probability Ratio} (LPR) for the FMAT.
#'
#' @param fmat A data.table (of new class \code{fmat})
#' returned by \code{\link{FMAT_run}}.
## @param digits Number of decimal places of output. Defaults to \code{3}.
#' @param ... Other arguments (currently not used).
#'
#' @return
#' A data.table of summarized results.
#'
#' @seealso
#' \code{\link{FMAT_run}}
#'
#' @export
summary.fmat = function(fmat, ...) {
  type = attr(fmat, "type")
  gvars.1 = c("model", "query", "M_pair",
              "TARGET", "T_pair", "T_word",
              "ATTRIB", "A_pair", "A_word")
  grouping.vars = intersect(names(fmat), gvars.1)
  M_word = T_word = A_word = MASK = TARGET = ATTRIB = prop = LPR = NULL

  dt = fmat[, .(
    MASK = paste(MASK[1], "-", MASK[2]),
    M_contr = paste(M_word[1], "-", M_word[2]),
    LPR = log(prop[1]) - log(prop[2])
  ), keyby = grouping.vars]
  dt$MASK = as_factor(dt$MASK)
  dt$M_contr = as_factor(dt$M_contr)
  dt$M_pair = NULL

  if(type=="MT") {
    if(nlevels(dt$TARGET)==2) {
      dt = dt[, .(
        TARGET = paste(TARGET[1], "-", TARGET[2]),
        T_contr = paste(T_word[1], "-", T_word[2]),
        LPR = LPR[1] - LPR[2]
      ), keyby = c("model", "query", "MASK", "M_contr", "T_pair")]
      dt$TARGET = as_factor(dt$TARGET)
      dt$T_contr = as_factor(dt$T_contr)
      dt$T_pair = NULL
    }
  }

  if(type=="MA") {
    if(nlevels(dt$ATTRIB)==2) {
      dt = dt[, .(
        ATTRIB = paste(ATTRIB[1], "-", ATTRIB[2]),
        A_contr = paste(A_word[1], "-", A_word[2]),
        LPR = LPR[1] - LPR[2]
      ), keyby = c("model", "query", "MASK", "M_contr", "A_pair")]
      dt$ATTRIB = as_factor(dt$ATTRIB)
      dt$A_contr = as_factor(dt$A_contr)
      dt$A_pair = NULL
    }
  }

  if(type=="MTA") {
    dt = dt[, .(
      LPR = mean(LPR)
    ), keyby = c("model", "query", "MASK", "M_contr",
                 "TARGET", "T_word", "ATTRIB")]
    dt = dt[, .(
      ATTRIB = paste(ATTRIB[1], "-", ATTRIB[2]),
      LPR = LPR[1] - LPR[2]
    ), keyby = c("model", "query", "MASK", "M_contr",
                 "TARGET", "T_word")]
  }

  return(dt)
}


