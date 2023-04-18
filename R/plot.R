#' @export
plot.statnn <-
  function(x, which = c(1L:ncol(x$X)), conf_int = FALSE, alpha = 0.05, B = x$B,
           method = c("deltamethod", "mlesim"),
           add_rug = TRUE,
           caption = lapply(
             1:ncol(x$X),
             function(iter) {
               paste0(
                 "Covariate-Effect Plot for ",
                 colnames(x$X)[iter]
               )
             }
           ),
           ylim = NULL, xlim = NULL, x_axis_l = 100,
           sub.caption = NULL, main = "",
           ask = prod(graphics::par("mfcol")) < length(which) && grDevices::dev.interactive(), ...,
           label.pos = c(4, 2), cex.caption = 1, cex.oma.main = 1.25) {
    if (!inherits(x, "statnn")) {
      stop("use only with \"statnn\" objects")
    }
    
    if (!is.numeric(which) || any(which < 1) || any(which > ncol(x$X))) {
      stop(sprintf("'which' must be in 1:%s", ncol(x$X)))
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
    
    show <- rep(FALSE, ncol(x$X))
    show[which] <- TRUE
    
    if (is.null(xlim)) {
      xlim <- vector("list", length = ncol(x$X))
      
      for (i in 1:ncol(x$X)) {
        xlim[[i]] <- c(min(x$X[, i]), max(x$X[, i]))
      }
      
    } else {
      xlim <- lapply(1:ncol(x$X), \(x) xlim)
    }
    
    cov_effs <- vector("list", length = ncol(x$X))
    conf_val <- vector("list", length = ncol(x$X))
    for (i in 1:ncol(x$X)) {
      if (show[i]) {
        
        
        
        cov_effs[[i]] <- pce(x$weights, x$X, x$n_nodes, i,
                             x_r = xlim[[i]], len = x_axis_l)
        
      }
      
      
      if (conf_int == TRUE && show[i]) {
        if (method[1] == "mlesim") {
          conf_val[[i]] <- mlesim(
            W = x$weights, X = x$X, y = x$y, q = x$n_nodes, ind = i,
            FUN = pce, B = B, alpha = alpha,
            x_r = xlim[[i]],
            len = x_axis_l, lambda = x$lambda, response = x$response
          )
        } else if (method[1] == "deltamethod") {
          conf_val[[i]] <- delta_method(
            W = x$weights, X = x$X, y = x$y, q = x$n_nodes,
            ind = i, FUN = pce,
            alpha = alpha, x_r = xlim[[i]],
            len = x_axis_l, lambda = x$lambda, response = x$response
          )
        }
      }
    }
    
    
    xaxis <- lapply(1:ncol(x$X), function(x) seq(xlim[[x]][1], xlim[[x]][2],
                                                 length.out = x_axis_l))
    
    xaxis <- lapply(1:ncol(x$X), 
                    function(x) {
                      if(min(xaxis[[x]]) == 0 & max(xaxis[[x]]) == 1)
                        xaxis[[x]] = 1 
                      else 
                        xaxis[[x]]
                    }) 
    
    labs <- colnames(x$X)
    
    one.fig <- prod(graphics::par("mfcol")) == 1
    if (ask) {
      oask <- grDevices::devAskNewPage(TRUE)
      on.exit(grDevices::devAskNewPage(oask))
    }
    ## ---------- Do the individual plots : ----------
    y_lim_user <- ylim
    for (i in 1:ncol(x$X)) {
      if (show[i]) {
        if (is.null(y_lim_user)) {
          ylim <- range(c(cov_effs[[i]], conf_val[[i]]), na.rm = TRUE)
          if (ylim[1] > 0) ylim[1] <- 0 else if (ylim[2] < 0) ylim[2] <- 0
        }
        
        grDevices::dev.hold()
        plot(xaxis[[i]], cov_effs[[i]],
             xlab = labs[i], ylab = "Effect", main = main,
             ylim = ylim, type = "n", ...
        )
        if (length(xaxis[[i]]) > 1) {
          graphics::lines(xaxis[[i]], cov_effs[[i]], ...)
        } else {
          graphics::points(xaxis[[i]], cov_effs[[i]], pch = 19, ...)
        }
        
        if (add_rug == TRUE) {
          graphics::rug(x$X[, i], quiet = TRUE)
        }
        if (conf_int == TRUE) {
          if (length(xaxis[[i]]) > 1) {
            graphics::polygon(c(rev(xaxis[[i]]), xaxis[[i]]),
                    c(rev(conf_val[[i]]$upper), conf_val[[i]]$lower),
                    col = 'grey80', border = NA)
            graphics::lines(xaxis[[i]], conf_val[[i]]$upper, lty = 2, col = 1, ...)
            graphics::lines(xaxis[[i]], conf_val[[i]]$lower, lty = 2, col = 1, ...)
            graphics::lines(xaxis[[i]], cov_effs[[i]], ...)
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
