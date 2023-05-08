#' Fits various tracks (different random starting values) and chooses best model
#'
#' Fits n_init tracks with different initial values and decides on best model
#' based on information criteria.
#'
#' @param x Matrix of covariates
#' @param y Vector of response
#' @param q Number of hidden nodes
#' @param n_init Number of random initialisations (tracks)
#' @param inf_crit Information criterion: `"BIC"` (default), `"AIC"` or
#'  `"AICc"`
#' @param lambda Ridge penalty
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @param unif Random initial values max value
#' @param maxit Maximum number of iterations for nnet (default = 100)
#' @param pkg Package for fitting neural network. One of `nnet` (default) or
#' `torch`
#' @param ... additional argument for nnet
#' @return The best model from the different initialisations
#' @export
nn_fit <- function(x, y, q, n_init, inf_crit = "BIC", lambda = 0,
                   response = "continuous", unif = 3, maxit = 1000,
                   pkg = "nnet", ...) {
  
  if (pkg == "nnet") {
    
    nn <- nn_fit_nnet(x = x, y = y, q = q, n_init = n_init, inf_crit = inf_crit,
                      lambda = lambda, response = response, unif = unif,
                      maxit = maxit, ...)
    
  } else if (pkg == "torch") {
    
    nn <- nn_fit_torch(x = x, y = y, q = q, n_init = n_init, inf_crit = inf_crit,
                       lambda = lambda, response = response, unif = unif,
                       maxit = maxit, ...)
    
  } else {
    stop(sprintf(
      "Error: %s not recognised as available package. Please choose nnet or torch",
      pkg
    ))
  }
  nn$x  <- x
  nn$y <- y
  return(nn)
}

#' Fits various tracks (different random starting values) and chooses best model
#' using nnet
#'
#' Fits n_init tracks with different initial values and decides on best model
#' based on information criteria.
#'
#' @param x Matrix of covariates
#' @param y Vector of response
#' @param q Number of hidden nodes
#' @param n_init Number of random initialisations (tracks)
#' @param inf_crit Information criterion: `"BIC"` (default), `"AIC"` or
#'  `"AICc"`
#' @param lambda Ridge penalty
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @param unif Random initial values max value
#' @param maxit maximum number of iterations for nnet (default = 100)
#' @param ... additional argument for nnet
#' @return The best model from the different tracks
#' @export
nn_fit_nnet <- function(x, y, q, n_init, inf_crit = "BIC", lambda = 0,
                        response = "continuous", unif = 3, maxit = 1000, ...) {
  # Function with fits n_init tracks of model and finds best
  
  df <- data.frame(x, y)
  n <- nrow(x)
  p <- ncol(as.matrix(x)) # as.matrix() in case p = 1 (auto. becomes vector)
  
  k <- (p + 2) * q + 1
  
  colnames(df)[ncol(df)] <- "y"
  
  weight_matrix_init <- matrix(stats::runif(n_init * k, min = -unif, max = unif), ncol = k)
  
  weight_matrix <- matrix(rep(NA, n_init * k), ncol = k)
  inf_crit_vec <- rep(NA, n_init)
  converge <- rep(NA, n_init)
  
  nn <- vector("list", length = n_init)
  
  if (response == "continuous") {
    linout <- TRUE
    entropy <- FALSE
  } else if (response == "binary") {
    linout <- FALSE
    entropy <- TRUE
  } else {
    stop(sprintf(
      "Error: %s not recognised as response. Please choose continuous or binary",
      response
    ))
  }
  
  for (iter in 1:n_init) {
    nn_model <- nnet::nnet(x = x, y = y, size = q, trace = FALSE,
                           linout = linout, entropy = entropy,
                           Wts = weight_matrix_init[iter, ], maxit = maxit,
                           decay = lambda, ...
    )
    
    weight_matrix[iter, ] <- nn_model$wts
    
    if (response == "continuous") {
      RSS <- nn_model$value
      sigma2 <- RSS / n
      
      log_likelihood <- (-n / 2) * log(2 * pi * sigma2) - RSS / (2 * sigma2)
      
      k_ic <- k + 1 # number of params for IC calculation
    } else if (response == "binary") {
      log_likelihood <- -nn_model$value
      k_ic <- k # number of params for IC calculation
    }
    
    inf_crit_vec[iter] <- ifelse(inf_crit == "AIC",
                                 (2 * (k_ic) - 2 * log_likelihood),
                                 ifelse(inf_crit == "BIC",
                                        (log(n) * (k_ic) - 2 * log_likelihood),
                                        ifelse(inf_crit == "AICc",
                                               (2 * (k_ic) * (n / (n - (k_ic) - 1)) - 2 * log_likelihood),
                                               NA)))
    converge[iter] <- nn_model$convergence
    nn[[iter]] <- nn_model
  }
  W_opt <- weight_matrix[which.min(inf_crit_vec), ]
  
  nn <- nn[[which.min(inf_crit_vec)]]
  
  return(list(
    "W_opt" = W_opt,
    "value" = min(inf_crit_vec),
    "inf_crit_vec" = inf_crit_vec,
    "weight_matrix" = weight_matrix,
    "convergence" = converge,
    "nn" = nn 
  ))
}

#' Fits various tracks (different random starting values) and chooses best model
#' using torch
#'
#' Fits n_init tracks with different initial values and decides on best model
#' based on information criteria.
#'
#' @param x Matrix of covariates
#' @param y Vector of response
#' @param q Number of hidden nodes
#' @param n_init Number of random initialisations (tracks)
#' @param inf_crit Information criterion: `"BIC"` (default), `"AIC"` or
#'  `"AICc"`
#' @param response `"continuous"` (default) or `"binary"`
#' @param unif Random initial values max value
#' @param maxit maximum number of iterations for nnet (default = 100)
#' @param lambda ridge penalty
#' @param min_delta tolerance for early stopping
#' @param patience patience for ealy stopping
#' @param batch_size batch size
#' @param ... additional argument for nnet
#' @return The best model from the different tracks
#' @export
nn_fit_torch <- function(x, y, q, n_init, inf_crit = "BIC",
                         response = "continuous", unif = 3, maxit = 1000,
                         lambda = 0, min_delta = 1.0e-8, patience = 10,
                         batch_size = 32, ...) {
  
  df <- data.frame(x, y)
  n <- nrow(x)
  p <- ncol(as.matrix(x)) 
  
  k <- (p + 2) * q + 1
  
  colnames(df)[ncol(df)] <- "y"
  
  weight_matrix_init <- matrix(stats::runif(n_init * k, min = -unif, max = unif),
                               ncol = k)
  
  weight_matrix <- matrix(rep(NA, n_init * k), ncol = k)
  inf_crit_vec <- rep(NA, n_init)
  converge <- rep(NA, n_init)
  
  nn <- vector("list", length = n_init)
  
  if (response == "continuous") {
    loss <- torch::nn_mse_loss()
    k_ic <- k + 1 # number of params for IC calculation
  } else if (response == "binary") {
    loss <- torch::nn_bce_loss()
    k_ic <- k # number of params for IC calculation
  } else {
    stop(sprintf(
      "Error: %s not recognised as task. Please choose regression or classification",
      response
    ))
  }
  
  for (iter in 1:n_init) {
    w_torch <- nnet_to_torch(weight_matrix_init[iter, ], p, q)
    
    self <- NULL
    
    modnn <- torch::nn_module(
      initialize = function(input_size) {
        self$hidden <- torch::nn_linear(input_size, q)
        self$hidden$register_parameter("weight", w_torch$hidden.weight)
        self$hidden$register_parameter("bias", w_torch$hidden.bias)
        self$activation <- torch::nn_sigmoid()
        self$output <- torch::nn_linear(q, 1)
        self$output$register_parameter("weight", w_torch$output.weight)
        self$output$register_parameter("bias", w_torch$output.bias)
      },
      forward = function(x) {
        x %>%
          self$hidden() %>%
          self$activation() %>%
          self$output()
      }
    )
    
    
    modnn <- modnn %>%
      luz::setup(
        loss = loss,
        optimizer = torch::optim_rmsprop,
        metrics = list(luz::luz_metric_accuracy(), luz::luz_metric_mse())
      ) %>%
      luz::set_hparams(input_size = p) %>% 
      luz::set_opt_hparams(weight_decay = lambda)
    
    fitted <- modnn %>%
      luz::fit(
        data = list(as.matrix(x), y),
        epochs = maxit,
        verbose = FALSE,
        callbacks = list(luz::luz_callback_early_stopping(monitor = "train_loss", 
                                                          min_delta =  min_delta, 
                                                          patience = patience),
                         print_callback(iter = iter)),
        dataloader_options = list(batch_size = batch_size)
      )
    
    weight_matrix[iter, ] <- torch_to_nnet(fitted$model$parameters)
    
    log_likelihood <- nn_loglike(fitted, X = x, y = y, lambda = lambda,
                                 response = response)
    
    inf_crit_vec[iter] <- ifelse(inf_crit == "AIC",
                                 (2 * (k_ic) - 2 * log_likelihood),
                                 ifelse(inf_crit == "BIC",
                                        (log(n) * (k_ic) - 2 * log_likelihood),
                                        ifelse(inf_crit == "AICc",
                                               (2 * (k_ic) * (n / (n - (k_ic) - 1)) - 2 * log_likelihood),
                                               NA)))
    
    converge[iter] <- length(fitted$records$metrics$train) < maxit
    nn[[iter]] <- fitted
  }
  W_opt <- weight_matrix[which.min(inf_crit_vec), ]
  
  nn <- nn[[which.min(inf_crit_vec)]]
  
  return(list(
    "W_opt" = W_opt,
    "value" = min(inf_crit_vec),
    "inf_crit_vec" = inf_crit_vec,
    "weight_matrix" = weight_matrix,
    "convergence" = converge,
    "nn" = nn 
  ))
}
