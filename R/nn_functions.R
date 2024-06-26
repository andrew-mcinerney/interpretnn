#' Neural network prediction
#'
#'
#' @param X Data
#' @param W Weight vector
#' @param q Number of hidden nodes
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Prediction for given inputs
#' @export
nn_pred <- function(X, W, q, response = "continuous") {
  n <- nrow(X)
  p <- ncol(X)
  
  X <- as.matrix(X)
  
  k <- sum(c(p + 1, q + 1) * c(q, 1))
  
  layer_nodes <- c(0, cumsum(c(p + 1, q + 1) * c(q, 1)))
  
  if (length(W) == k) {
    
    X <- cbind(rep(1, n), X)
    
    temp <- X
    
    for(i in 1:length(q)) {
      
      h_input <- temp %*% t(matrix(W[(layer_nodes[i] + 1):layer_nodes[i + 1]],
                                   nrow = q[i], byrow = TRUE))
      
      h_act <- cbind(rep(1, n), sigmoid(h_input))
      
      temp <- h_act
    }
    
    
    if (response == "continuous") {
      y_hat <- h_act %*% 
        matrix(W[(layer_nodes[length(layer_nodes) - 1] + 1):
                   layer_nodes[length(layer_nodes)]],
               ncol = 1)
    } else if (response == "binary") {
      y_hat <- sigmoid(
        h_act %*% matrix(W[c((length(W) - q):length(W))], ncol = 1)
      )
    } else {
      stop(sprintf(
        "Error: %s not recognised as response. Please choose continuous or binary",
        response
      ))
    }
    
    return(y_hat)
  } else {
    stop(sprintf(
      "Error: Incorrect number of weights for NN structure. W should have
      %s weights (%s weights supplied).", k, length(W)
    ))
  }
}

#' Neural network loss
#'
#'
#' @param X Data
#' @param W Weight vector
#' @param y Response variable
#' @param q Number of hidden nodes
#' @param lambda Ridge peanlty
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return loss for given neural network
#' @export
nn_loss <- function(W, X, y, q, lambda = 0, response = "continuous") {
  pred <- nn_pred(X, W, q, response)
  
  val <- sum((pred - y)^2) + lambda * sum(W ^ 2)
  
  return(val)
}

#' Neural Network Normal Log-likelihood Value
#'
#'
#' @param object neural network object
#' @param X input data (required for keras)
#' @param y response variable (required for keras)
#' @param lambda Ridge penalty
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Log-Likelihood value
#' @export
nn_loglike <- function(object, X = NULL, y = NULL, lambda = 0,
                       response = "continuous") {
  
  if (!(response %in% c("continuous", "binary"))) {
    stop(sprintf(
      "Error: %s not recognised as response. Please choose continuous or binary",
      response
    ))
  }
  
  if (class(object)[1] == "nnet" | class(object)[1] == "nnet.formula") {
    
    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == nnet")
    }
    
    weights <- object$wts
    
    q <- object$n[2]
    
    y <- object$fitted.values + object$residuals

    
  } else if (class(object)[1] == "keras.engine.sequential.Sequential" | 
             class(object)[1] == "keras.engine.functional.Functional") {
    if (is.null(y)) {
      stop("Error: Argument y must not be NULL when class(object) == keras.engine.sequential.Sequential or 
           keras.engine.functional.Functional")
    }

    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == keras.engine.sequential.Sequential or 
           keras.engine.functional.Functional")
    }

    keras_weights <- keras::get_weights(object)

    weights <- c(
      as.vector(rbind(keras_weights[[2]], keras_weights[[1]])),
      c(keras_weights[[4]], keras_weights[[3]])
    )
    
    q <-  ncol(keras_weights[[1]])
    
  } else if (class(object)[1] == "nn") {
    
    weights <- c(as.vector(object$weights[[1]][[1]]),
                        object$weights[[1]][[2]])
    
    q <- sapply(object$weights[[1]], ncol)
    
    q <- q[-length(q)]
    
    y <- object$response
    
    X <- object$covariate
    
  
  } else if (class(object)[1] == "luz_module_fitted") {
    if (is.null(y)) {
      stop("Error: Argument y must not be NULL when class(object) == luz_module_fitted")
    }
    
    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == luz_module_fitted")
    }
    
    luz_weights <- object$model$parameters
    
    weights <- c(
      as.vector(t(cbind(as.matrix(luz_weights$hidden.bias),
                        as.matrix(luz_weights$hidden.weight)))),
      cbind(as.matrix(luz_weights$output.bias),
            as.matrix(luz_weights$output.weight))
    )
    
    q <- length(luz_weights$output.weight)
    
  } else if (class(object)[1] == "ANN") {
    if (is.null(y)) {
      stop("Error: Argument y must not be NULL when class(object) == ANN")
    }
    
    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == ANN")
    }
    
    weights<- c(as.vector(t(cbind(object$Rcpp_ANN$getParams()[[2]][[1]],
                                     object$Rcpp_ANN$getParams()[[1]][[1]]))),
                   as.vector(t(cbind(object$Rcpp_ANN$getParams()[[2]][[2]],
                                     object$Rcpp_ANN$getParams()[[1]][[2]]))))
    
    q <- object$Rcpp_ANN$getMeta()$num_nodes[-c(1, length(object$Rcpp_ANN$getMeta()$num_nodes))]
    
  }
  
  if (response == "continuous") {
    n <- nrow(X)
    
    RSS <- sum((nn_pred(X, weights, q) - y)^2)
    
    sigma2 <- RSS / n
    
    log_like <- (-n / 2) * log(2 * pi * sigma2) - RSS / (2 * sigma2) -
      lambda * sum(weights ^ 2)
  } else if (response == "binary") {
    log_like <- y * log(nn_pred(X, weights, q,
                                response = "binary")) +
      (1 - y) * log(1 - nn_pred(X, weights, q,
                                response = "binary")) -
      lambda * sum(weights ^ 2)
  }

  return(log_like)
}


#' Neural network gradient for one observation
#'
#'
#' @param W Weight vector
#' @param X Input vector
#' @param q Number of hidden nodes
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Log-Likelihood value
#' @noRd
nn_grad_1 <- function(W, X, q, response = "continuous") {
  
  p <- length(X)
  
  X1 <- c(1, X)
  
  g <- W[(((p + 1) * q + 1):((p + 2) * q + 1))]
  wt <- matrix(W[1:((p + 1) * q)], nrow = p + 1)
  
  H <- cbind(1, 1 / (1 + exp(-X1%*%wt)))
  
  if (response == "continuous") {
    d1 <- X1 %*% (t(g[-1]) * sigmoid(X1%*%wt) * (1 - sigmoid(X1%*%wt)))
    d2 <- H
  } else if (response == "binary") {
    d1 <- X1 %*% (t(g[-1]) * sigmoid(X1%*%wt) * (1 - sigmoid(X1%*%wt))) *
      as.vector(sigmoid(H %*% g) * (1 - sigmoid(H %*% g)))
    d2 <- t(H) %*% (sigmoid(H %*% g) * (1 - sigmoid(H %*% g)))
  }
  
  grad <- c(as.vector(d1), d2) 
  
  return(grad)
}

#' Neural network gradient
#'
#'
#' @param W Weight vector
#' @param X Input vector
#' @param q Number of hidden nodes
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Log-Likelihood value
#' @export
nn_grad <- function(W, X, q, response = "continuous") {
  
  X <- as.matrix(X)
  
  grad <- t(apply(X, 1, \(x) nn_grad_1(W, x, q, response = response)))
  
  return(grad)
  
}