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

#' Fits various tracks (different random starting values) and chooses best model
#'
#' Fits n_init tracks with different initial values and decides on best model
#' based on information criteria.
#'
#' @param X Matrix of covariates
#' @param y Vector of response
#' @param q Number of hidden nodes
#' @param n_init Number of random initialisations (tracks)
#' @param inf_crit Information criterion: `"BIC"` (default), `"AIC"` or
#'  `"AICc"`
#' @param task `"regression"` (default) or `"classification"`
#' @param unif Random initial values max value
#' @param maxit maximum number of iterations for nnet (default = 100)
#' @param ... additional argument for nnet
#' @return The best model from the different tracks
#' @export
nn_fit_tracks <- function(X, y, q, n_init, inf_crit = "BIC",
                          task = "regression", unif = 3, maxit = 1000, ...) {
  # Function with fits n_init tracks of model and finds best
  
  df <- data.frame(X, y)
  n <- nrow(X)
  p <- ncol(as.matrix(X)) # as.matrix() in case p = 1 (auto. becomes vector)
  
  k <- (p + 2) * q + 1
  
  colnames(df)[ncol(df)] <- "y"
  
  weight_matrix_init <- matrix(stats::runif(n_init * k, min = -unif, max = unif), ncol = k)
  
  weight_matrix <- matrix(rep(NA, n_init * k), ncol = k)
  inf_crit_vec <- rep(NA, n_init)
  converge <- rep(NA, n_init)
  
  if (task == "regression") {
    linout <- TRUE
    entropy <- FALSE
  } else if (task == "classification") {
    linout <- FALSE
    entropy <- TRUE
  } else {
    stop(sprintf(
      "Error: %s not recognised as task. Please choose regression or classification",
      task
    ))
  }
  
  for (iter in 1:n_init) {
    nn_model <- nnet::nnet(y ~ .,
                           data = df, size = q, trace = FALSE,
                           linout = linout, entropy = entropy,
                           Wts = weight_matrix_init[iter, ], maxit = maxit, ...
    )
    
    weight_matrix[iter, ] <- nn_model$wts
    
    if (task == "regression") {
      RSS <- nn_model$value
      sigma2 <- RSS / n
      
      log_likelihood <- (-n / 2) * log(2 * pi * sigma2) - RSS / (2 * sigma2)
    } else if (task == "classification") {
      log_likelihood <- -nn_model$value
    }
    
    inf_crit_vec[iter] <- ifelse(inf_crit == "AIC",
                                 (2 * (k + 1) - 2 * log_likelihood),
                                 ifelse(inf_crit == "BIC",
                                        (log(n) * (k + 1) - 2 * log_likelihood),
                                        ifelse(inf_crit == "AICc",
                                               (2 * (k + 1) * (n / (n - (k + 1) - 1)) - 2 * log_likelihood),
                                               NA
                                        )
                                 )
    )
    converge[iter] <- nn_model$convergence
  }
  W_opt <- weight_matrix[which.min(inf_crit_vec), ]
  
  return(list(
    "W_opt" = W_opt,
    "value" = min(inf_crit_vec),
    "inf_crit_vec" = inf_crit_vec,
    "converge" = converge,
    "weight_matrix" = weight_matrix
  ))
}

#' Neural Network Normal Log-likelihood Value
#'
#'
#' @param object neural network object
#' @param X input data (required for keras)
#' @param y response variable (required for keras)
#' @param lambda Ridge peanlty
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Log-Likelihhod value
#' @export
nn_loglike <- function(object, X = NULL, y = NULL, lambda = 0,
                       response = "continuous") {
  
  if (!(response %in% c("continuous", "binary"))) {
    stop(sprintf(
      "Error: %s not recognised as response. Please choose continuous or binary",
      response
    ))
  }
  
  if (class(object)[1] == "nnet") {
    
    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == nnet")
    }
    
    weights <- object$wts
    
    q <- object$n[2]
    
    y <- object$fitted.values + object$residuals

    
  } else if (class(object)[1] == "keras.engine.sequential.Sequential") {
    if (is.null(y)) {
      stop("Error: Argument y must not be NULL when class(object) == keras.engine.sequential.Sequential")
    }

    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == keras.engine.sequential.Sequential")
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
      stop("Error: Argument y must not be NULL when class(object) == keras.engine.sequential.Sequential")
    }
    
    if (is.null(X)) {
      stop("Error: Argument X must not be NULL when class(object) == keras.engine.sequential.Sequential")
    }
    
    luz_weights <- object$model$parameters
    
    weights <- c(
      as.vector(t(cbind(as.matrix(luz_weights$hidden.bias),
                        as.matrix(luz_weights$hidden.weight)))),
      cbind(as.matrix(luz_weights$output.bias),
            as.matrix(luz_weights$output.weight))
    )
    
    q <- length(luz_weights$output.weight)
    
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
