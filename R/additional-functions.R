#' Fits various tracks (different random starting values) and chooses best model
#' using torch
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
nn_fit_torch <- function(X, y, q, n_init, inf_crit = "BIC",
                         response = "continuous", unif = 3, maxit = 1000,
                         lambda = 0, min_delta = 1.0e-8, patience = 10,
                         batch_size = 32, ...) {
  
  df <- data.frame(X, y)
  n <- nrow(X)
  p <- ncol(as.matrix(X)) 
  
  k <- (p + 2) * q + 1
  
  colnames(df)[ncol(df)] <- "y"
  
  weight_matrix_init <- matrix(stats::runif(n_init * k, min = -unif, max = unif),
                               ncol = k)
  
  weight_matrix <- matrix(rep(NA, n_init * k), ncol = k)
  inf_crit_vec <- rep(NA, n_init)
  converge <- rep(NA, n_init)
  
  if (response == "continuous") {
    loss <- torch::nn_mse_loss()
  } else if (response == "binary") {
    loss <- torch::nn_bce_loss()
  } else {
    stop(sprintf(
      "Error: %s not recognised as task. Please choose regression or classification",
      response
    ))
  }
  
  for (iter in 1:n_init) {
    w_torch <- nnet_to_torch(weight_matrix_init[iter, ], p, q)
    
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
        data = list(as.matrix(X), y),
        epochs = maxit,
        verbose = FALSE,
        callbacks = list(luz::luz_callback_early_stopping(monitor = "train_loss", 
                                                     min_delta =  min_delta, 
                                                     patience = patience),
                         print_callback(iter = iter)),
        dataloader_options = list(batch_size = batch_size)
      )
    
    weight_matrix[iter, ] <- torch_to_nnet(fitted$model$parameters)
    
    log_likelihood <- nn_loglike(fitted, X = X, y = y, lambda = lambda,
                                 response = response)
    
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
    converge[iter] <- length(fitted$records$metrics$train) < maxit
  }
  W_opt <- weight_matrix[which.min(inf_crit_vec), ]
  
  return(list(
    "W_opt" = W_opt,
    "value" = min(inf_crit_vec),
    "inf_crit_vec" = inf_crit_vec,
    "weight_matrix" = weight_matrix,
    "convergence" = converge
  ))
}

#' torch weights to nnet weights
#'
#'
#' @param torch_w weights in torch format
#' @return weights in nnet format
#' @export
torch_to_nnet <- function(torch_w) {
  nnet_w <- c(
    as.vector(t(cbind(as.matrix(torch_w$hidden.bias),
                      as.matrix(torch_w$hidden.weight)))),
    cbind(as.matrix(torch_w$output.bias),
          as.matrix(torch_w$output.weight))
  )
  
  return(nnet_w)
}

#' nnet weights to torch weights
#'
#'
#' @param nnet_w weights in nnet format
#' @param p number of inputs
#' @param q number of hidden nodes
#' @return weights in torch format
#' @export
nnet_to_torch <- function(nnet_w, p, q) {
  hidden_b_ind <- sapply(1:q, function(x) (x - 1) * (p + 1) + 1)
  hidden_w_ind <- c(1:((p + 1) * q))[-hidden_b_ind]
  
  output_b_ind <- (p + 1) * q + 1
  output_w_ind <- ((p + 1) * q + 2):((p + 2) * q + 1)
  
  torch_w <- vector("list")
  torch_w$hidden.weight <- matrix(nnet_w[hidden_w_ind], nrow = q, byrow = TRUE)
  torch_w$hidden.bias <- nnet_w[hidden_b_ind]
  torch_w$output.weight <- matrix(nnet_w[output_w_ind], ncol = 2)
  torch_w$output.bias <- nnet_w[output_b_ind]
  
  torch_w <- lapply(torch_w, torch::torch_tensor, requires_grad = TRUE)
  
  return(torch_w)
}

print_callback <- luz::luz_callback(
  name = "print_callback",
  initialize = function(iter) {
    self$iter <- iter
  },
  on_fit_end = function(iter) {
    cat("Iteration ", self$iter, "Done \n")
  }
)