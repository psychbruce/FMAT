% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/FMAT.R
\name{BERT_download}
\alias{BERT_download}
\title{Download BERT models.}
\usage{
BERT_download(models = NULL)
}
\arguments{
\item{models}{Model names at \href{https://huggingface.co/models}{HuggingFace}.

For a full list of available BERT models, see
\url{https://huggingface.co/models?pipeline_tag=fill-mask&library=transformers}}
}
\value{
No return value.
}
\description{
Download and save BERT models to local cache folder "\%USERPROFILE\%/.cache/huggingface".
}
\examples{
\dontrun{
model.names = c("bert-base-uncased", "bert-base-cased")
BERT_download(model.names)

BERT_download()  # check downloaded models
}

}
\seealso{
\code{\link{FMAT_load}}
}