## News

Thanks. This is the first (re)submission of this package to CRAN.

I have revised it according to Victoria Wimmer's suggestions:

1.  In the DESCRIPTION file, I added (1) single quotes for
software names and API and (2) necessary references for
'BERT' (Devlin et al., 2018) <arXiv:1810.04805> and
'Hugging Face' <https://huggingface.co/models?pipeline_tag=fill-mask>.
Note that this package goes with a new method described in
my working paper which has not been published yet.
In future versions, should my research article be accepted
or publicly available, I will add it as a reference in the
DESCRIPTION file.

2.  Added return value description for dot-.Rd.

3.  Replaced all \dontrun{} with \donttest{} in examples.

4.  Added () behind function names in DESCRIPTION and omitted the quotes.

## Test environments

-   Windows 11 (local installation), R 4.3.0
-   Mac OS 13.3 (check_mac_release), R 4.3.0

## Package check results

passing `devtools::check_win_devel()`

## R CMD check results

passing (0 errors | 0 warnings | 0 notes)
