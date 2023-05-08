#' Plot neural network architecture 
#'
#' \code{plotnn} It is designed for an inspection of the weights for objects of
#' class \code{statnn}.
#'
#'
#' @param x an object of class \code{statnnet}
#' @param rep repetition of the neural network. If rep="best", the repetition
#' with the smallest error will be plotted. If not stated all repetitions will
#' be plotted, each in a separate window.
#' @param x.entry x-coordinate of the entry layer. Depends on the arrow.length
#' in default.
#' @param x.out x-coordinate of the output layer.
#' @param radius radius of the neurons.
#' @param arrow.length length of the entry and out arrows.
#' @param intercept a logical value indicating whether to plot the intercept.
#' @param intercept.factor x-position factor of the intercept. The closer the
#' factor is to 0, the closer the intercept is to its left neuron.
#' @param information a logical value indicating whether to add the error and
#' steps to the plot.
#' @param information.pos y-position of the information.
#' @param col.entry.synapse color of the synapses leading to the input neurons.
#' @param col.entry color of the input neurons.
#' @param col.hidden color of the neurons in the hidden layer.
#' @param col.hidden.synapse color of the weighted synapses.
#' @param col.out color of the output neurons.
#' @param col.out.synapse color of the synapses leading away from the output
#' neurons.
#' @param col.intercept color of the intercept.
#' @param col.sig.synapse color of the significant synapses.
#' @param col.insig.synapse color of the insignificant synapses.
#' @param col.sig.node color of the significant input nodes.
#' @param col.insig.node color of the insignificant input nodes.
#' @param fontsize fontsize of the text.
#' @param dimension size of the plot in inches.
#' @param show.weights a logical value indicating whether to print the
#' calculated weights above the synapses.
#' @param file a character string naming the plot to write to. If not stated,
#' the plot will not be saved.
#' @param rounding number of decimal places to round values.
#' @param alpha significane level.
#' @param lambda ridge penalty.
#' @param \dots arguments to be passed to methods, such as graphical parameters
#' (see \code{\link{par}}).
#'
#' @return plot of weights and their significance
#' @export
plotnn <-
  function (x, rep = NULL, x.entry = NULL, x.out = NULL, radius = 0.15,
            arrow.length = 0.2, intercept = FALSE, intercept.factor = 0.4,
            information = FALSE, information.pos = 0.1, col.entry.synapse = "black",
            col.entry = "black", col.hidden = "black", col.hidden.synapse = "black",
            col.out = "black", col.out.synapse = "black", col.intercept = "black",
            col.sig.synapse = "black", col.insig.synapse = "lightgrey",
            col.sig.node = "black", col.insig.node = "lightgrey",
            fontsize = 12, dimension = 6, show.weights = FALSE, file = NULL,
            rounding = 3, alpha = 0.05, lambda = 0,
            ...)
  {
    net <- x
    p <- net$n_inputs
    q <- net$n_nodes
    net$wts <- list(list(
      matrix(net$weights[1:((p + 1) * q)],
             nrow = p + 1, ncol = q),
      matrix(net$weights[((p + 1) * q + 1):length(net$weights)],
             nrow = q + 1, ncol = 1)))
    p_values_w <- net$wald_sp$p_value
    sig_w <- p_values_w < alpha
    
    p_values_n <- net$wald$p_value
    sig_n <- p_values_n < alpha
    
    sig_w <- ifelse(sig_w, col.sig.synapse, col.insig.synapse)
    synapse.col <- list(
      matrix(sig_w[1:((p + 1) * q)],
             nrow = p + 1, ncol = q),
      matrix(sig_w[((p + 1) * q + 1):length(sig_w)],
             nrow = q + 1, ncol = 1))
    sig_n <- ifelse(sig_n, col.sig.node, col.insig.node)
    
    if (is.null(net$weights))
      stop("weights were not calculated")
    if (!is.null(file) && !is.character(file))
      stop("'file' must be a string")
    if (is.null(rep)) {
      for (i in 1:length(net$wts)) {
        if (!is.null(file))
          file.rep <- paste(file, ".", i, sep = "")
        else file.rep <- NULL
        # grDevices::dev.new()
        plotnn(net, rep = i, x.entry, x.out, radius, arrow.length,
               intercept, intercept.factor, information, information.pos,
               col.entry.synapse, col.entry, col.hidden, col.hidden.synapse,
               col.out, col.out.synapse, col.intercept, col.sig.synapse,
               col.insig.synapse, col.sig.node, col.insig.node, fontsize,
               dimension, show.weights, file.rep, rounding = rounding,
               alpha = alpha, lambda = lambda, ...)
      }
    }
    else {
      if (is.character(file) && file.exists(file))
        stop(sprintf("%s already exists", sQuote(file)))
      # result.matrix <- t(net$result.matrix)
      # if (rep == "best")
      #   rep <- as.integer(which.min(result.matrix[, "error"]))
      if (rep > length(net$wts))
        stop("'rep' does not exist")
      weights <- net$wts[[rep]]
      if (is.null(x.entry))
        x.entry <- 0.5 - (arrow.length/2) * length(weights)
      if (is.null(x.out))
        x.out <- 0.5 + (arrow.length/2) * length(weights)
      width <- max(x.out - x.entry + 0.2, 0.8) * 8
      radius <- radius/dimension
      entry.label <- colnames(net$X)
      out.label <- colnames(net$y)
      neuron.count <- array(0, length(weights) + 1)
      neuron.count[1] <- nrow(weights[[1]]) - 1
      neuron.count[2] <- ncol(weights[[1]])
      x.position <- array(0, length(weights) + 1)
      x.position[1] <- x.entry
      x.position[length(weights) + 1] <- x.out
      if (length(weights) > 1)
        for (i in 2:length(weights)) {
          neuron.count[i + 1] <- ncol(weights[[i]])
          x.position[i] <- x.entry + (i - 1) * (x.out -
                                                  x.entry)/length(weights)
        }
      y.step <- 1/(neuron.count + 1)
      y.position <- array(0, length(weights) + 1)
      y.intercept <- 1 - 2 * radius
      information.pos <- min(min(y.step) - 0.1, 0.2)
      if (length(entry.label) != neuron.count[1]) {
        if (length(entry.label) < neuron.count[1]) {
          tmp <- NULL
          for (i in 1:(neuron.count[1] - length(entry.label))) {
            tmp <- c(tmp, "no name")
          }
          entry.label <- c(entry.label, tmp)
        }
      }
      if (length(out.label) != neuron.count[length(neuron.count)]) {
        if (length(out.label) < neuron.count[length(neuron.count)]) {
          tmp <- NULL
          for (i in 1:(neuron.count[length(neuron.count)] -
                       length(out.label))) {
            tmp <- c(tmp, "no name")
          }
          out.label <- c(out.label, tmp)
        }
      }
      grid::grid.newpage()
      for (k in 1:length(weights)) {
        for (i in 1:neuron.count[k]) {
          y.position[k] <- y.position[k] + y.step[k]
          y.tmp <- 0
          for (j in 1:neuron.count[k + 1]) {
            y.tmp <- y.tmp + y.step[k + 1]
            result <- calculate.delta(c(x.position[k],
                                        x.position[k + 1]), c(y.position[k], y.tmp),
                                      radius)
            x <- c(x.position[k], x.position[k + 1])
            y <- c(y.position[k], y.tmp)
            grid::grid.lines(x = x,
                             y = y,
                             # arrow = grid::arrow(length = grid::unit(0.15, "cm"),
                             #                     type = "closed"),
                             gp = grid::gpar(fill =
                                               synapse.col[[k]][
                                                 neuron.count[k] - i + 2,
                                                 neuron.count[k + 1] - j + 1],
                                             col = synapse.col[[k]][
                                               neuron.count[k] - i + 2,
                                               neuron.count[k + 1] - j + 1], ...))
            if (show.weights)
              draw.text(label = round(weights[[k]][neuron.count[k] -
                                                     i + 2, neuron.count[k + 1] - j + 1],
                                      digits = rounding),
                        x = c(x.position[k],
                              x.position[k + 1]),
                        y = c(y.position[k],
                              y.tmp),
                        xy.null = 1.25 * result,
                        color = col.hidden.synapse,
                        fontsize = fontsize - 2, ...)
          }
          if (k == 1) {
            # grid::grid.lines(x = c((x.position[1] - arrow.length),
            #                        x.position[1] - radius), y = y.position[k],
            #                  arrow = grid::arrow(length = grid::unit(0.15, "cm"),
            #                                      type = "closed"), gp = grid::gpar(fill = col.entry.synapse,
            #                                                                        col = col.entry.synapse, ...))
            draw.text(label = entry.label[(neuron.count[1] +
                                             1) - i],
                      x = c((x.position -  radius), x.position[1] - radius),
                      y = c(y.position[k]- 0.5 * radius ,
                            y.position[k]- 0.5 * radius),
                      xy.null = c(0, 0),
                      color = col.entry.synapse,
                      alignment = c("right", "bottom"),
                      fontsize = fontsize, ...)
            grid::grid.circle(x = x.position[k], y = y.position[k],
                              r = radius, gp = grid::gpar(fill = "white", col = rev(sig_n)[i],
                                                          ...))
            # grid::grid.text(paste0("X", i), x = x.position[k], y = y.position[k],
            #                 gp = grid::gpar(col = col.entry.synapse,
            #                                 fontsize = fontsize - 2, ...))
          }
          else {
            grid::grid.circle(x = x.position[k], y = y.position[k],
                              r = radius, gp = grid::gpar(fill = "white", col = col.hidden,
                                                          ...))
          }
        }
      }
      out <- length(neuron.count)
      for (i in 1:neuron.count[out]) {
        y.position[out] <- y.position[out] + y.step[out]
        # grid::grid.lines(x = c(x.position[out] + radius,
        #                        x.position[out] + arrow.length),
        #                  y = y.position[out],
        #                  arrow = grid::arrow(length = grid::unit(0.15, "cm"),
        #                                      type = "closed"),
        #                  gp = grid::gpar(fill = col.out.synapse,
        #                                  col = col.out.synapse, ...))
        
        draw.text(label = out.label[(neuron.count[out] +
                                       1) - i],
                  x = c((x.position[out] + radius), x.position[out] + arrow.length),
                  y = c(y.position[out] - 0.5 * radius, y.position[out] - 0.5 * radius),
                  xy.null = c(0, 0),
                  color = col.out.synapse,
                  fontsize = fontsize,
                  ...)
        grid::grid.circle(x = x.position[out], y = y.position[out],
                          r = radius, gp = grid::gpar(fill = "white", col = col.out,
                                                      ...))
      }
      if (intercept) {
        intercept.labels <- c("intercept", "")
        for (k in 1:length(weights)) {
          y.tmp <- 0
          x.intercept <- (x.position[k + 1] - x.position[k]) *
            intercept.factor + x.position[k]
          for (i in 1:neuron.count[k + 1]) {
            y.tmp <- y.tmp + y.step[k + 1]
            result <- calculate.delta(c(x.intercept, x.position[k +
                                                                  1]), c(y.intercept, y.tmp), radius)
            x <- c(x.intercept, x.position[k + 1])
            y <- c(y.intercept, y.tmp + result[2])
            grid::grid.lines(x = x,
                             y = y,
                             # arrow = grid::arrow(length = grid::unit(0.15, "cm"),
                             #                     type = "closed"),
                             gp = grid::gpar(fill = synapse.col[[k]][1,
                                                                     neuron.count[k + 1] - i + 1],
                                             col = synapse.col[[k]][1,
                                                                    neuron.count[k + 1] - i + 1], ...))
            xy.null <- cbind(x.position[k + 1] - x.intercept -
                               2 * result[1], -(y.tmp - y.intercept + 2 *
                                                  result[2]))
            if (show.weights)
              draw.text(label = round(weights[[k + 1]][i, ],
                                      digits = rounding),
                        x = c(x.intercept, x.position[k + 1]),
                        y = c(y.intercept, y.tmp),
                        xy.null = xy.null,
                        color = col.intercept,
                        alignment = c("right", "bottom"),
                        fontsize = fontsize - 2,
                        ...)
          }
          grid::grid.circle(x = x.intercept, y = y.intercept,
                            r = radius, gp = grid::gpar(fill = "white", col = col.intercept,
                                                        ...))
          
          draw.text(label = intercept.labels[k],
                    x = c((x.intercept - radius), x.position[k + 1]),
                    y = c(y.intercept, y.intercept),
                    xy.null = c(0, 0),
                    color = col.intercept,
                    fontsize = fontsize,
                    alignment = c("right", "bottom"),
                    ...)
        }
      }
      if (information)
        grid::grid.text(paste("RSS: ", round(net$val, rounding),
                              "   BIC: ", round(net$BIC, rounding), sep = ""),
                        x = 0.5, y = information.pos,
                        just = "bottom", gp = grid::gpar(fontsize = fontsize +
                                                           2, ...))
      if (!is.null(file)) {
        weight.plot <- grDevices::recordPlot()
        save(weight.plot, file = file)
      }
    }
  }
calculate.delta <-
  function (x, y, r)
  {
    delta.x <- x[2] - x[1]
    delta.y <- y[2] - y[1]
    x.null <- r/sqrt(delta.x^2 + delta.y^2) * delta.x
    if (y[1] < y[2])
      y.null <- -sqrt(r^2 - x.null^2)
    else if (y[1] > y[2])
      y.null <- sqrt(r^2 - x.null^2)
    else y.null <- 0
    c(x.null, y.null)
  }
draw.text <-
  function (label, x, y, xy.null = c(0, 0), color, alignment = c("left",
                                                                 "bottom"), ...)
  {
    x.label <- x[1] + xy.null[1]
    y.label <- y[1] - xy.null[2]
    x.delta <- x[2] - x[1]
    y.delta <- y[2] - y[1]
    angle = atan(y.delta/x.delta) * (180/pi)
    if (angle < 0)
      angle <- angle + 0
    else if (angle > 0)
      angle <- angle - 0
    if (is.numeric(label))
      label <- round(label, 5)
    vp <- grid::viewport(x = x.label, y = y.label, width = 0, height = ,
                         angle = angle, name = "vp1", just = alignment)
    grid::grid.text(label, x = 0, y = grid::unit(0.75, "mm"), just = alignment,
                    gp = grid::gpar(col = color, ...), vp = vp)
  }
