library(nnet)
library(keras)
library(neuralnet)
library(ANN2)

library(torch)
library(luz)

# prep data ---------------------------------------------------------------

set.seed(1)
n <- 500
p <- 4
q <- 2
K <- (p + 2) * q + 1

X <- matrix(rnorm(p * n), ncol = p)

W <- nnic::my_runif(K, 3, 1)

y <- nnic::nn_pred(X, W, q) + rnorm(n)

# nnet --------------------------------------------------------------------

nn <- nnet(X, y, size = q, linout = TRUE, trace = FALSE)


# using statnn function

stnn <- statnn(nn, X)
summary(stnn)


# keras -------------------------------------------------------------------

model <- keras_model_sequential()
model %>%
  layer_dense(units = q, activation = "sigmoid", input_shape = p) %>%
  layer_dense(units = 1, activation = "linear")

summary(model)


model %>% compile(
  loss = loss_mean_squared_error(),
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# need to add weight decay at each layer and for weights and biases seperately

# model1 <- keras_model_sequential()
# model1 %>%
#   layer_dense(units = q, activation = "sigmoid", input_shape = p,
#               kernel_regularizer = regularizer_l2(l = 0.01), 
#               bias_regularizer = regularizer_l2(l = 0.01)) %>%
#   layer_dense(units = 1, activation = "sigmoid",
#               kernel_regularizer = regularizer_l2(l = 0.01), 
#               bias_regularizer = regularizer_l2(l = 0.01))

nn_keras <- model %>%
  fit(X, y, epochs = 100, batch_size = nrow(X))

keras_weights <- model %>% get_weights()

W_keras <- c(
  as.vector(rbind(keras_weights[[2]], keras_weights[[1]])),
  c(keras_weights[[4]], keras_weights[[3]])
)

sum((nn_pred(X, W_keras, q) - y)^2)


# using statnn function

stnn <- statnn(model, X, y)
summary(stnn)

# neuralnet ---------------------------------------------------------------

neural_model <- neuralnet(y ~ ., data = data.frame(X, y), hidden = q,
                          err.fct = "sse", act.fct = "logistic",
                          linear.output = TRUE)

summary(neural_model)
neural_model
print(neural_model)

neural_model$result.matrix[1, ] * 2
neural_weights <- c(as.vector(neural_model$weights[[1]][[1]]),
                    neural_model$weights[[1]][[2]])


yhat <- statnn::nn_pred(X, neural_weights, q)
neural_model$net.result[[1]] - yhat

sum((yhat - y)^2)

# using statnn function

stnn <- statnn(neural_model)
summary(stnn)

plot(stnn)


# ANN2 --------------------------------------------------------------------

ann_model <- neuralnetwork(X, y, q, regression = TRUE, standardize = FALSE,
                           loss.type = "squared", activ.functions = "sigmoid",
                           batch.size = nrow(X), val.prop = 0, n.epochs = 100000)


ann_weights <- c(as.vector(t(cbind(ann_model$Rcpp_ANN$getParams()[[2]][[1]],
                             ann_model$Rcpp_ANN$getParams()[[1]][[1]]))),
           as.vector(t(cbind(ann_model$Rcpp_ANN$getParams()[[2]][[2]],
                             ann_model$Rcpp_ANN$getParams()[[1]][[2]]))))

# ann_model$Rcpp_ANN$getTrainHistory()$train_loss

sum((y - predict(ann_model, newdata = X)[[1]])^2) / nrow(X)

statnn::nn_pred(X, ann_weights, q) - predict(ann_model, newdata = X)[[1]]



# torch -------------------------------------------------------------------


modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, q)
    self$activation <- nn_sigmoid()
    self$output <- nn_linear(q, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden() %>%
      self$activation() %>%
      self$output()
  }
)


modnn <- modnn %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams(input_size = p)

# weight decay
# 
# modnn <- modnn %>%
#   setup(
#     loss = nn_mse_loss(),
#     optimizer = optim_rmsprop,
#     metrics = list(luz_metric_accuracy())
#   ) %>%
#   set_hparams(input_size = p) %>% 
#   set_opt_hparams(weight_decay = 0.01)

# binary response
#
# modnn1 <- nn_module(
#   initialize = function(input_size) {
#     self$hidden <- nn_linear(input_size, q)
#     self$activation <- nn_sigmoid()
#     self$output <- nn_linear(q, 1)
#   },
#   forward = function(x) {
#     x %>%
#       self$hidden() %>%
#       self$activation() %>%
#       self$output() %>% 
#       self$activation()
#   }
# )
# 
# 
# modnn1 <- modnn1 %>%
#   setup(
#     loss = nn_bce_loss(),
#     optimizer = optim_rmsprop,
#     metrics = list(luz_metric_accuracy())
#   ) %>%
#   set_hparams(input_size = p) %>% 
#   set_opt_hparams(weight_decay = 0.01)
# 
# 
# fitted1 <- modnn1 %>%
#   fit(
#     data = list(X, as.numeric(y > 0)),
#     epochs = 100
#   )
# 
# 
# W_torch <- c(
#   as.vector(t(cbind(as.matrix(fitted1$model$parameters$hidden.bias),
#                     as.matrix(fitted1$model$parameters$hidden.weight)))),
#   cbind(as.matrix(fitted1$model$parameters$output.bias),
#         as.matrix(fitted1$model$parameters$output.weight))
# )
# 
# sum(as.numeric(y > 0) * log(nn_pred(X, W_torch, q, response = "binary")) +
#       (1 - as.numeric(y > 0)) * log(1 - nn_pred(X, W_torch, q, response = "binary")))
# 
# fitted1$records$metrics$train[[100]]$loss * length(y)


fitted <- modnn %>%
  fit(
    data = list(X, y),
    epochs = 100
  )



W_torch <- c(
  as.vector(t(cbind(as.matrix(fitted$model$parameters$hidden.bias),
                    as.matrix(fitted$model$parameters$hidden.weight)))),
  cbind(as.matrix(fitted$model$parameters$output.bias),
                    as.matrix(fitted$model$parameters$output.weight))
)

sum((nn_pred(X, W_torch, q) - y)^2) / nrow(X)

stnn <- statnn(fitted, X, y)
summary(stnn, wald_single_par = TRUE)
