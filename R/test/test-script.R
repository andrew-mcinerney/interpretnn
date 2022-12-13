library(nnet)
library(keras)
library(neuralnet)

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


yhat <- nn_pred(X, neural_weights, q)
neural_model$net.result[[1]] - yhat

sum((yhat - y)^2)

# using statnn function

stnn <- statnn(neural_model)
summary(stnn)

plot(stnn)
