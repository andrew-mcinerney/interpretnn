#' @export
plot.statnn <-
  function(x, which = c(1L:ncol(x$X)), x_axis_r = c(-3, 3), x_axis_l = 101,
           conf_int = FALSE, alpha = 0.05, B = x$B, method = c(
             "mlesim",
             "deltamethod"
           ),
           rug_flag = TRUE,
           caption = lapply(
             1:ncol(x$X),
             function(iter) {
               paste0(
                 "Covariate-Effect Plot for ",
                 colnames(x$X)[iter]
               )
             }
           ),
           ylim = NULL,
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

    cov_effs <- lapply(
      1:ncol(x$X),
      function(iter) {
        pdp_effect(x$weights, x$X, x$n_nodes,
          iter,
          x_r = x_axis_r,
          len = x_axis_l
        )
      }
    )

    conf_val <- vector("list", length = ncol(x$X))
    for (i in 1:ncol(x$X)) {
      if (conf_int == TRUE && show[i]) {
        if (method[1] == "mlesim") {
          conf_val[[i]] <- mlesim(
            W = x$weights, X = x$X, y = x$y, q = x$n_nodes, ind = i,
            FUN = pdp_effect, B = B, alpha = alpha,
            x_r = x_axis_r,
            len = x_axis_l
          )
        } else if (method[1] == "deltamethod") {
          conf_val[[i]] <- delta_method(
            W = x$weights, X = x$X, y = x$y, q = x$n_nodes,
            ind = i, FUN = pdp_effect,
            alpha = alpha, x_r = x_axis_r,
            len = x_axis_l
          )
        }
      }
    }


    xaxis <- seq(x_axis_r[1], x_axis_r[2], length.out = x_axis_l)

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
        plot(xaxis, cov_effs[[i]],
          xlab = labs[i], ylab = "Effect", main = main,
          ylim = ylim, type = "n", ...
        )
        graphics::lines(xaxis, cov_effs[[i]], ...)
        if (rug_flag == TRUE) {
          graphics::rug(x$X[, i], quiet = TRUE)
        }
        if (conf_int == TRUE) {
          graphics::lines(xaxis, conf_val[[i]]$upper, lty = 2, col = 2, ...)
          graphics::lines(xaxis, conf_val[[i]]$lower, lty = 2, col = 2, ...)
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
