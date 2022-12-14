#' @export
print.statnn <- function(x, ...) {
  # cat("Call (nnet):\n")
  # print(x$call)
  cat("Call (statnn):\n")
  print(x$cl)
  cat("\n")
  cat("Model Architecture: ", x$n_inputs, "-",
      paste0(x$n_nodes, collapse = "-"), "-", "1", " network", sep = ""
  )
  cat(" with", x$n_param, "weights\n")
}

#' @export
coef.statnn <- function(object, ...) {
  weights <- object$weights
  
  layer_sizes <- c(object$n_inputs, object$n_nodes, 1)
  
  
  weight_names_in <- rep(c("b0", colnames(object$X)), times = object$n_nodes[1])
  weight_names_out <- c()
  for (i in 1:length(object$n_nodes)) {
    weight_names_in <- c(weight_names_in, 
                         rep(c(paste0("b", i), 
                               paste0("h", i, 1:object$n_nodes[i])),
                             layer_sizes[i + 2]))
    
    weight_names_out <- c(weight_names_out, 
                          rep(paste0("h", i, 1:object$n_nodes[i]),
                              each = layer_sizes[i] + 1))
  }
  weight_names_out <- c(weight_names_out, rep("y", 
                                              object$n_nodes[length(object$n_nodes)] + 1))
  
  weight_names <- paste(weight_names_in, weight_names_out, sep = "->")                      
  
  
  names(weights) <- weight_names
  return(weights)
}

#' @export
summary.statnn <- function(object, ...) {
  
  n_nodes <- c(object$n_inputs, object$n_nodes, 1)
  
  nconn <- cumsum(c(rep(0, times = n_nodes[1] + 2),
                    unlist(mapply(function (x, y) rep(x, times = y),
                                  n_nodes[-length(n_nodes)] + 1, n_nodes[-1]))))

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
  cat("Number of hidden nodes:", paste(x$n_nodes, collapse = ", "), "\n")
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
  # lapply(
  #   split(wts, rep(1:(x$n_inputs + x$n_nodes + 2), diff(x$nconn))),
  #   function(x) print(x, quote = FALSE)
  # )
  print(wts, quote = FALSE)
}
