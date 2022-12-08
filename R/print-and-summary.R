#' @export
print.statnn <- function(x, ...) {
  # cat("Call (nnet):\n")
  # print(x$call)
  cat("Call (statnn):\n")
  print(x$cl)
  cat("\n")
  cat("Model Architecture: ", x$n_inputs, "-", x$n_nodes, "-", "1", " network",
    sep = ""
  )
  cat(" with", x$n_param, "weights\n")
}

#' @export
coef.statnn <- function(object, ...) {
  weights <- object$weights

  wm <- c(
    "b",
    paste(colnames(object$X), sep = ""),
    paste("h", seq_len(object$n_nodes), sep = ""),
    as.character(ifelse(!is.null(names(object$y)), names(object$y), "y"))
  )

  conn <- c(
    rep(0:object$n_inputs, times = object$n_nodes), 0,
    (object$n_inputs + 1):(object$n_inputs + object$n_nodes)
  )

  nunits <- object$n_inputs + object$n_nodes + 2

  nconn <- c(
    rep(0, times = object$n_inputs + 2),
    seq(object$n_inputs + 1, object$n_nodes * (object$n_inputs + 1),
      by = (object$n_inputs + 1)
    ),
    object$n_nodes * (object$n_inputs + 2) + 1
  )

  names(weights) <- apply(
    cbind(
      wm[1 + conn],
      wm[1 + rep(1:nunits - 1, diff(nconn))]
    ),
    1,
    function(x) paste(x, collapse = "->")
  )
  return(weights)
}

#' @export
summary.statnn <- function(object, ...) {
  nconn <- c(
    rep(0, times = object$n_inputs + 2),
    seq(object$n_inputs + 1, object$n_nodes * (object$n_inputs + 1),
      by = (object$n_inputs + 1)
    ),
    object$n_nodes * (object$n_inputs + 2) + 1
  )

  object$nconn <- nconn

  covariates <- colnames(object$X)

  coefdf <- data.frame(
    Covariate = covariates,
    Estimate = object$eff[, 1],
    Std.Error = object$eff[, 2],
    Break1 = rep("|", length(object$wald$chisq)),
    Wald.chi = object$wald$chisq,
    Wald.p.value = object$wald$p_value
  )

  colnames(coefdf)[1] <- ""
  colnames(coefdf)[3] <- "Std. Error"
  colnames(coefdf)[4] <- "|"
  colnames(coefdf)[5] <- "  X^2"
  colnames(coefdf)[6] <- "Pr(> X^2)"

  object$coefdf <- coefdf

  Signif <- stats::symnum(object$wald$p_value,
    corr = FALSE, na = FALSE,
    cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
    symbols = c("***", "**", "*", ".", " ")
  )
  object$coefdf$`Pr(> X^2)` <- paste(
    formatC(object$coefdf$`Pr(> X^2)`, format = "e", digits = 2),
    format(Signif)
  )

  class(object) <- c("summary.statnn", class(object))
  return(object)
}

#' @export
print.summary.statnn <- function(x, ...) {
  ## code to get wald in right place (may need editing later)

  fdf <- format(x$coefdf)
  strings <- apply(x$coefdf, 2, function(x) unlist(format(x)))[1, ]
  rowname <- format(rownames(fdf))[[1]]
  strings <- c(rowname, strings)
  widths <- nchar(strings)
  names <- c("", colnames(x))
  widths <- pmax(nchar(strings), nchar(names))
  csum <- sum(widths[2:5] + 1) - 1 # first 5 aren't associated with Wald
  csum <- csum + mean(widths[6:7]) # to be centered above columns

  ##

  # cat("Call (nnet):\n")
  # print(x$call)
  cat("Call (statnnet):\n")
  print(x$cl)
  cat("\n")
  cat("Number of input nodes:", x$n_inputs, "\n")
  cat("Number of hidden nodes:", x$n_nodes, "\n")
  cat("\n")
  cat("BIC:", x$BIC, "\n")
  cat("\n")
  cat("Coefficients:\n")
  writeLines(paste(c(rep(" ", csum - 3), "Wald"), collapse = ""))
  print(x$coefdf,
    right = TRUE, na.print = "NA",
    digits = max(3L, getOption("digits") - 2L), row.names = FALSE
  )

  Signif <- stats::symnum(x$wald$p_value,
    corr = FALSE, na = FALSE,
    cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
    symbols = c("***", "**", "*", ".", " ")
  )
  if ((w <- getOption("width")) < nchar(sleg <- attr(
    Signif,
    "legend"
  ))) {
    sleg <- strwrap(sleg, width = w - 2, prefix = "  ")
  }
  cat("---\nSignif. codes:  ", sleg, sep = "", fill = w +
    4 + max(nchar(sleg, "bytes") - nchar(sleg)))
  cat("\n")
  cat("Weights:\n")
  wts <- format(round(coef.statnn(x), 2))
  lapply(
    split(wts, rep(1:(x$n_inputs + x$n_nodes + 2), diff(x$nconn))),
    function(x) print(x, quote = FALSE)
  )
}
