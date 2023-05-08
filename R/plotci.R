#' Plot Wald Confidence Intervals
#'
#' \code{plotci} It is designed for an inspection of the confidence intervals 
#' for the weights for objects of
#' class \code{statnn}.
#'
#'
#' @param object an object of class \code{statnnet}.
#' @param alpha significane level.
#' @param which index of plots to be displayed.
#' @param which_params index of weights to be displayed.
#' @param colour colour of confidence intervals.
#' @param ask ask before displaying each plot.
#' @param caption caption for each plot.
#' @param \dots arguments to be passed to methods, such as graphical parameters
#' (see \code{\link{par}}).
#'
#' @return plot of weights and their significance
#' @export
plotci <- function(object, alpha = 0.05, which = c(1L:object$n_inputs),
                   which_params = c(1L:object$n_nodes),
                   colour = 1,
                   ask = prod(graphics::par("mfcol")) < length(which) &&
                     grDevices::dev.interactive(),
                   caption = lapply(
                     1:ncol(object$X),
                     function(iter) {
                       paste0(
                         "Confidence Intervals for ",
                         colnames(object$X)[iter]
                       )
                     }
                   ), ...) {
  
  if (length(which_params) < 2) {
    stop("Error: which_params must be of length 2 or more.")
  }
  
  sp_ci <- wald_single_parameter(object$X, object$y, object$weights,
                                 object$n_nodes, object$lambda,
                                 object$response, alpha)$ci
  
  p <- object$n_inputs
  q <- object$n_nodes
  weights <- object$weights
  n <- object$n
  
  vc <- VC(weights, object$X, object$y, q, lambda = object$lambda,
           response = object$response)
  
  show <- rep(FALSE, ncol(object$X))
  show[which] <- TRUE
  
  one.fig <- prod(graphics::par("mfcol")) == 1
  if (ask) {
    oask <- grDevices::devAskNewPage(TRUE)
    on.exit(grDevices::devAskNewPage(oask))
  }
  
  ## ---------- Do the individual plots : ----------
  for (i in 1:ncol(object$X)) {
    if (show[i]) {
      ind_vec <- sapply(
        X = 1:q[1],
        FUN = function(x) (x - 1) * (p + 1) + 1 + i
      )
      
      theta_x <- weights[ind_vec]
      vc_x <- vc[ind_vec, ind_vec]
      
      plot_points <- t(utils::combn(ind_vec, 2))
      
      ind_points <- matrix(match(plot_points, ind_vec), ncol = 2)
      
      plot_points <- plot_points[
        apply(ind_points, 1, function (x) all(x %in% which_params)), ,
        drop = FALSE]
      
      ind_points <- ind_points[
        apply(ind_points, 1, function (x) all(x %in% which_params)), , 
        drop = FALSE]
      
      xlabs <- paste0("w", ind_points[, 1])
      ylabs <- paste0("w", ind_points[, 2])
      
      for(j in 1:nrow(plot_points)) {
        
        vc_x_temp <- vc_x[ind_points[j, ], ind_points[j, ]]
        theta_x_temp <- theta_x[ind_points[j, ]]
        
        radius <- sqrt(stats::qchisq(1 - alpha, df = 2))
        ellipse <- car::ellipse(shape = vc_x_temp, center = theta_x_temp, radius = radius, 
                                draw = FALSE)
        
        xlim <- c(min(c(0, sp_ci[plot_points[j, 1], 1], ellipse[, 1])),
                  max(c(0, sp_ci[plot_points[j, 1], 2], ellipse[, 1])))
        ylim <- c(min(c(0, sp_ci[plot_points[j, 2], 1], ellipse[, 2])),
                  max(c(0,sp_ci[plot_points[j, 2], 2], ellipse[, 2])))
        
        grDevices::dev.hold()
        
        plot(weights[plot_points[j, 1]], weights[plot_points[j, 2]],
             xlim = xlim, ylim = ylim,
             xlab = xlabs[j], ylab = ylabs[j], main = caption[[i]],
             type = "p", col = colour, ...)
        graphics::abline(h = 0)
        graphics::abline(v = 0)
        graphics::lines(ellipse, col = colour, ...)
        
        graphics::rect(xleft = sp_ci[plot_points[j, 1], 1], ybottom = sp_ci[plot_points[j, 2], 1],
                       xright =  sp_ci[plot_points[j, 1], 2], ytop = sp_ci[plot_points[j, 2], 2],
                       lty = "dashed", border = colour, ...)
        
        # grDevices::dev.flush()
      }
    }
  }
  invisible()
}
