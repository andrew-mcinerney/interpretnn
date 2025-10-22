#' @export
plot.interpretnn <-
  function(x, which = c(1L:ncol(X)), conf_int = FALSE, alpha = 0.05, B = x$B,
           method = c("deltamethod", "mlesim"),
           add_rug = TRUE,
           caption = lapply(
             1:ncol(X),
             function(iter) {
               paste0(
                 "Covariate-Effect Plot for ",
                 colnames(X)[iter]
               )
             }
           ),
           ylim = NULL, xlim = NULL, x_axis_l = 100,
           sub.caption = NULL, main = "",
           ask = prod(graphics::par("mfcol")) < length(which) && grDevices::dev.interactive(), ...,
           label.pos = c(4, 2), cex.caption = 1, cex.oma.main = 1.25) {
    
    X <- stats::model.matrix(x$formula, data = x$data)[, -1, drop = FALSE] 
    y <- as.matrix(stats::model.extract(stats::model.frame(x$formula, data = x$data),
                                        "response"), ncol = 1)
    
    if (!inherits(x, "interpretnn")) {
      stop("use only with \"interpretnn\" objects")
    }
    
    if (!is.numeric(which) || any(which < 1) || any(which > ncol(X))) {
      stop(sprintf("'which' must be in 1:%s", ncol(X)))
    }
    
    if (conf_int == TRUE && is.null(alpha)) {
      stop("'alpha' must be not be NULL when 'conf_int == TRUE'")
    } else if (conf_int == TRUE && (alpha < 0 || alpha > 1 || length(alpha) != 1)) {
      stop("'alpha' must be a value between 0 and 1")
    } else if (conf_int == TRUE && !(method %in% c(
      "mlesim",
      "deltamethod"
    ))) {
      stop("'method' must be one of 'mlesim' or 'deltamethod'")
    }
    
    getCaption <- function(k) { # allow caption = "" , plotmath etc
      if (length(caption) < k) NA_character_ else grDevices::as.graphicsAnnot(caption[[k]])
    }
    
    show <- rep(FALSE, ncol(X))
    show[which] <- TRUE
    
    if (is.null(xlim)) {
      xlim <- vector("list", length = ncol(X))
      
      for (i in 1:ncol(X)) {
        xlim[[i]] <- c(min(X[, i]), max(X[, i]))
      }
      
    } else {
      xlim <- lapply(1:ncol(X), \(x) xlim)
    }
    
    cov_effs <- vector("list", length = ncol(X))
    conf_val <- vector("list", length = ncol(X))
    for (i in 1:ncol(X)) {
      if (show[i]) {
        
        
        
        cov_effs[[i]] <- pce(x$weights, X, x$n_nodes, i,
                             x_r = xlim[[i]], len = x_axis_l)
        
      }
      
      
      if (conf_int == TRUE && show[i]) {
        if (method[1] == "mlesim") {
          conf_val[[i]] <- mlesim(
            W = x$weights, X = X, y = y, q = x$n_nodes, ind = i,
            FUN = pce, B = B, alpha = alpha,
            x_r = xlim[[i]],
            len = x_axis_l, lambda = x$lambda, response = x$response
          )
        } else if (method[1] == "deltamethod") {
          conf_val[[i]] <- delta_method(
            W = x$weights, X = X, y = y, q = x$n_nodes,
            ind = i, FUN = pce,
            alpha = alpha, x_r = xlim[[i]],
            len = x_axis_l, lambda = x$lambda, response = x$response
          )
        }
      }
    }
    
    
    xaxis <- lapply(1:ncol(X), function(x) seq(xlim[[x]][1], xlim[[x]][2],
                                               length.out = x_axis_l))
    
    xaxis <- lapply(1:ncol(X), 
                    function(x) {
                      if(min(xaxis[[x]]) == 0 & max(xaxis[[x]]) == 1)
                        xaxis[[x]] = 1 
                      else 
                        xaxis[[x]]
                    }) 
    
    labs <- colnames(X)
    
    one.fig <- prod(graphics::par("mfcol")) == 1
    if (ask) {
      oask <- grDevices::devAskNewPage(TRUE)
      on.exit(grDevices::devAskNewPage(oask))
    }
    ## ---------- Do the individual plots : ----------
    y_lim_user <- ylim
    for (i in 1:ncol(X)) {
      if (show[i]) {
        if (is.null(y_lim_user)) {
          ylim <- range(c(cov_effs[[i]]$eff, conf_val[[i]]), na.rm = TRUE)
          if (ylim[1] > 0) ylim[1] <- 0 else if (ylim[2] < 0) ylim[2] <- 0
        }
        
        grDevices::dev.hold()
        plot(xaxis[[i]], cov_effs[[i]]$eff,
             xlab = labs[i], ylab = "Effect", main = main,
             ylim = ylim, type = "n", ...
        )
        if (length(xaxis[[i]]) > 1) {
          graphics::lines(xaxis[[i]], cov_effs[[i]]$eff, ...)
        } else {
          graphics::points(xaxis[[i]], cov_effs[[i]]$eff, pch = 19, ...)
        }
        
        if (add_rug == TRUE) {
          graphics::rug(X[, i], quiet = TRUE)
        }
        if (conf_int == TRUE) {
          if (length(xaxis[[i]]) > 1) {
            graphics::polygon(c(rev(xaxis[[i]]), xaxis[[i]]),
                              c(rev(conf_val[[i]]$upper), conf_val[[i]]$lower),
                              col = 'grey80', border = NA)
            graphics::lines(xaxis[[i]], conf_val[[i]]$upper, lty = 2, col = 1, ...)
            graphics::lines(xaxis[[i]], conf_val[[i]]$lower, lty = 2, col = 1, ...)
            graphics::lines(xaxis[[i]], cov_effs[[i]]$eff, ...)
          } else {
            graphics::arrows(xaxis[[i]], conf_val[[i]]$lower,
                             xaxis[[i]], conf_val[[i]]$upper, 
                             length = 0.05, angle = 90, code = 3)
          }
        }
        if (one.fig) {
          graphics::title(sub = sub.caption, ...)
        }
        graphics::mtext(getCaption(i), 3, 0.25, cex = cex.caption)
        graphics::abline(h = 0, lty = 3, col = "gray")
        grDevices::dev.flush()
      }
    }
    if (!one.fig && graphics::par("oma")[3L] >= 1) {
      graphics::mtext(sub.caption, outer = TRUE, cex = cex.oma.main)
    }
    invisible()
  }
