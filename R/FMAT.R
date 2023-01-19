#### Initialize ####


#' @import stringr
#' @import data.table
#' @importFrom forcats as_factor
#' @importFrom bruceR cc Glue Print dtime
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
    cat("\n")
    cli::cli_alert_success("
    Packages also loaded: {.pkg data.table, stringr}
    ")
    cat("\n")
  }
}


#### Basic ####


#' A wrapper of \code{list()}.
#' @export
. = list

#' @importFrom bruceR cc
#' @export
bruceR::cc

#' @importFrom PsychWordVec text_init
#' @export
PsychWordVec::text_init

#' @importFrom PsychWordVec text_model_download
#' @export
PsychWordVec::text_model_download

#' @importFrom PsychWordVec text_model_remove
#' @export
PsychWordVec::text_model_remove

#' @importFrom PsychWordVec text_unmask
#' @export
PsychWordVec::text_unmask

text_initialized = function() {
  error = TRUE
  try({
    text::textModels()
    error = FALSE
  }, silent=TRUE)
  if(error) PsychWordVec::text_init()
}


#### FMAT ####


# query = "[MASK] is ABC."
# expand_pair(query, .(High=cc("high, strong"), Low=cc("low, weak")))
# expand_pair(query, .(H="high", M="medium", L="low"))
# X = .(Flower=cc("rose, iris, lily"), Pos=cc("health, happiness, love, peace"))
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


#' Produce a data.table of queries and variables for the FMAT.
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
#' \code{\link{FMAT}}
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
#'   MASK = .(Male=cc("man, boy"), Female=cc("woman, girl")),
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
    query = "[MASK] is {TARGET}.",
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
    dq = map_query(query, expand_pair, MASK)
  } else if(length(TARGET)>0 & length(ATTRIB)==0) {
    # Only TARGET
    dq = append_X(map_query(query, expand_pair, MASK),
                  TARGET, "TARGET")
  } else if(length(TARGET)==0 & length(ATTRIB)>0) {
    # Only ATTRIB
    dq = append_X(map_query(query, expand_pair, MASK),
                  ATTRIB, "ATTRIB")
  } else if(length(TARGET)>0 & length(ATTRIB)>0) {
    # Both TARGET and ATTRIB
    dq = map_query(query, function(q, target, attrib) {
      rbind(
        expand_full(q, c(target[1], attrib[1])),
        expand_full(q, c(target[1], attrib[2])),
        expand_full(q, c(target[2], attrib[1])),
        expand_full(q, c(target[2], attrib[2]))
      )
    }, TARGET, ATTRIB)
  }
  if(any(str_count(query, "\\[MASK\\]") > 1))
    dq$unmask.id = as.integer(unmask.id)
  if(length(query) > 1)
    dq = cbind(data.table(qid = as.factor(as.numeric(dq$query))), dq)
  return(dq[order(query)])
}


#' Combind multiple query data.tables and renumber query ids.
#'
#' @param ... Query data.tables returned from \code{\link{FMAT_query}}.
#'
#' @return
#' A data.table of queries and variables.
#'
#' @seealso
#' \code{\link{FMAT}}
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
  dqs = rbind(...)
  if("qid" %in% names(dqs)) dqs$qid = NULL
  dqs = cbind(data.table(qid = as.factor(as.numeric(dqs$query))), dqs)
  return(dqs)
}


#' Run the FMAT on multiple models.
#' @export
FMAT_run = function(model, data.query, parallel = FALSE) {
  res = lapply(model, function(model.i) {
    Print(model.i)
  })
  return(res)
}


#' The Fill-Mask Association Test (\code{FMAT_query()} & \code{FMAT_run()}).
#'
#' @inheritParams FMAT_query
#' @inheritParams FMAT_run
#'
#' @seealso
#' \code{\link{FMAT_query}}
#'
#' \code{\link{FMAT_query_bind}}
#'
#' \code{\link{FMAT_run}}
#'
#' @examples
#' models = c("bert-base-uncased",
#'            "bert-base-cased")
#'
#' FMAT(
#'   models,
#'   "The {TARGET} has a [MASK] association with {ATTRIB}.",
#'   MASK = .(H="high", L="low"),
#'   TARGET = .(Flower=cc("rose, iris, lily"),
#'              Insect=cc("ant, cockroach, spider")),
#'   ATTRIB = .(Pos=cc("health, happiness, love, peace"),
#'              Neg=cc("death, sickness, hatred, disaster"))
#' )
#'
#' @export
FMAT = function(
    model = "bert-base-uncased",
    query = "[MASK] is {TARGET}.",
    MASK = .(),
    TARGET = .(),
    ATTRIB = .(),
    unmask.id = 1,
    parallel = FALSE
) {
  # Many Queries
  dq = FMAT_query(query, MASK, TARGET, ATTRIB)

  # Many Models
  data = FMAT_run(model, dq, parallel)

  return(data)
}

