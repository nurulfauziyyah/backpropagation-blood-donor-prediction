# 1. Load Library
library(readxl)
library(ROSE)
library(smotefamily)
library(ggplot2)
library(pROC)

# 2. Persiapan Data
x <- read_excel("C:/Users/LENOVO2/OneDrive - uny.ac.id/Statistics/STAT(6)/Jaringan Syaraf Tiruan/blood.xlsx")
table(x$donated)
x$donated <- as.numeric(as.character(x$donated))  # pastikan biner
normalize <- function(x) (x - min(x)) / (max(x) - min(x))

# 3. Split Train/Test Sebelum SMOTE & ROSE
set.seed(42)
n <- nrow(x)
train_idx <- sample(1:n, size = 0.8 * n)
x_train <- x[train_idx, ]
x_test <- x[-train_idx, ]

# 4. ROSE di data training
x_train_bal <- x_train 
x_train_bal$donated <- factor(x_train_bal$donated)
set.seed(123)
colnames(x_train_bal) <- make.names(colnames(x_train_bal))
rose_result <- ROSE(donated ~ ., data = x_train_bal, seed = 1)$data
rose_result$donated <- as.numeric(as.character(rose_result$donated))

# 4. SMOTE di data training
smote_result <- SMOTE(X = x_train[, 1:4], target = x_train$donated, K = 5, dup_size = 3)
x_train_smote <- smote_result$data
x_train_smote$donated <- as.numeric(as.character(x_train_smote$class))
x_train_smote$class <- NULL

# 5. Normalisasi ROSE pakai min-max dari data training
min_train <- sapply(rose_result[, 1:4], min)
max_train <- sapply(rose_result[, 1:4], max)
x_train_norm <- as.data.frame(mapply(function(col, minv, maxv) (col - minv)/(maxv - minv),
                                     rose_result[, 1:4], min_train, max_train))
x_train_norm$donated <- rose_result$donated
x_test_norm <- as.data.frame(mapply(function(col, minv, maxv) (col - minv)/(maxv - minv),
                                    x_test[, 1:4], min_train, max_train))
x_test_norm$donated <- x_test$donated

# 5. Normalisasi SMOTE pakai min-max dari data training
min_train <- sapply(x_train_smote[, 1:4], min)
max_train <- sapply(x_train_smote[, 1:4], max)
x_train_norm <- as.data.frame(mapply(function(col, minv, maxv) (col - minv)/(maxv - minv),
                                     x_train_smote[, 1:4], min_train, max_train))
x_train_norm$donated <- x_train_smote$donated
x_test_norm <- as.data.frame(mapply(function(col, minv, maxv) (col - minv)/(maxv - minv),
                                    x_test[, 1:4], min_train, max_train))
x_test_norm$donated <- x_test$donated

# 6. Konversi ke Matriks
X_train <- as.matrix(x_train_norm[, 1:4])
Y_train <- as.matrix(x_train_norm$donated)

X_test <- as.matrix(x_test_norm[, 1:4])
Y_test <- as.matrix(x_test_norm$donated)

# 7. Inisialisasi Parameter
input_size <- ncol(X_train)
hidden_size <- 5
output_size <- 1
set.seed(123)
w1 <- matrix(runif(input_size * hidden_size, -1, 1), nrow = input_size)
b1 <- runif(hidden_size, -1, 1)
w2 <- matrix(runif(hidden_size * output_size, -1, 1), nrow = hidden_size)
b2 <- runif(output_size, -1, 1)
learning_rate <- 0.01
epochs <- 1000
prev_loss <- Inf

# 8. Fungsi Aktivasi
sigmoid <- function(x) 1 / (1 + exp(-x))
sigmoid_deriv <- function(x) sigmoid(x) * (1 - sigmoid(x))
ReLU <- function(x) pmax(0, x)
ReLU_deriv <- function(x) ifelse(x > 0, 1, 0)

loss_history <- c()
acc_history <- c()

# 9. Training Loop
for (epoch in 1:epochs) {
  z1 <- X_train %*% w1 + matrix(b1, nrow = nrow(X_train), ncol = hidden_size, byrow = TRUE)
  a1 <- matrix(ReLU(z1), nrow = nrow(z1), ncol = ncol(z1))
  
  z2 <- a1 %*% w2 + matrix(b2, nrow = nrow(a1), ncol = output_size, byrow = TRUE)
  a2 <- matrix(sigmoid(z2), nrow = nrow(z2), ncol = ncol(z2))
  
  loss <- -mean(Y_train * log(a2 + 1e-8) + (1 - Y_train) * log(1 - a2 + 1e-8))
  
  delta2 <- (a2 - Y_train) * sigmoid_deriv(z2)
  dw2 <- t(a1) %*% delta2
  db2 <- colSums(delta2)
  
  delta1 <- (delta2 %*% t(w2)) * ReLU_deriv(z1)
  dw1 <- t(X_train) %*% delta1
  db1 <- colSums(delta1)
  
  w2 <- w2 - learning_rate * dw2
  b2 <- b2 - learning_rate * db2
  w1 <- w1 - learning_rate * dw1
  b1 <- b1 - learning_rate * db1
  
  # Simpan loss dan akurasi training setiap epoch
  loss_history <- c(loss_history, -loss)  # loss bernilai negatif karena BCE, dibalik
  pred_train <- ifelse(a2 >= 0.5, 1, 0)
  acc_epoch <- mean(pred_train == Y_train)
  acc_history <- c(acc_history, acc_epoch)
  
  if (epoch %% 100 == 0) {
    cat("Epoch:", epoch, "Loss:", loss, "\n")
  }
  if (abs(prev_loss - loss) < 1e-6) {
    cat("Loss konvergen di epoch", epoch, "\n")
    break
  }
  prev_loss <- loss
}

# 10. Evaluasi di Test Set
z1_test <- X_test %*% w1 + matrix(b1, nrow = nrow(X_test), ncol = hidden_size, byrow = TRUE)
a1_test <- matrix(ReLU(z1_test), nrow = nrow(z1_test), ncol = ncol(z1_test))
z2_test <- a1_test %*% w2 + matrix(b2, nrow = nrow(a1_test), ncol = output_size, byrow = TRUE)
a2_test <- matrix(sigmoid(z2_test), nrow = nrow(z2_test), ncol = ncol(z2_test))

predicted_test <- ifelse(a2_test >= 0.5, 1, 0)
true_test <- as.numeric(Y_test)

# 11. Confusion Matrix & Metrics
conf_matrix <- table(Predicted = predicted_test, Actual = true_test)
print(conf_matrix)

TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrix)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat(sprintf("Accuracy : %.3f\n", accuracy))
cat(sprintf("Precision: %.3f\n", precision))
cat(sprintf("Recall   : %.3f\n", recall))
cat(sprintf("F1 Score : %.3f\n", f1_score))

# Visualisasi Loss dan Accuracy
plot(loss_history, type = "l", col = "red", main = "Loss per Epoch", ylab = "Loss", xlab = "Epoch")
plot(acc_history, type = "l", col = "blue", main = "Accuracy per Epoch", ylab = "Accuracy", xlab = "Epoch")

# Visualisasi Confusion Matrix
conf_matrix <- table(Predicted = predicted_test, Actual = true_test)
conf_matrix_prop <- prop.table(conf_matrix, margin = 2)
cm_df <- as.data.frame(as.table(conf_matrix_prop))
colnames(cm_df) <- c("Predicted", "Actual", "Proportion")
cm_df$Predicted <- factor(cm_df$Predicted, levels = c(0, 1))
cm_df$Actual <- factor(cm_df$Actual, levels = c(0, 1))
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Proportion)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Proportion)), size = 6, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", name = "Proportion") +
  theme_minimal() +
  labs(title = "Normalized Confusion Matrix",
       x = "True Label", y = "Predicted Label") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 13))

# Visualisasi ROC object dari skor prediksi
roc_obj <- roc(true_test, as.numeric(a2_test))
auc_score <- auc(roc_obj)
plot(roc_obj, col = "blue", lwd = 2,
     main = paste("ROC Curve - AUC:", round(auc_score, 3)))
abline(a = 0, b = 1, lty = 2, col = "gray")

# Visualisasi Jaringan Syaraf Tiruan
n_input <- nrow(w1)     # 4 input
n_hidden <- ncol(w1)    # hidden_size
n_output <- ncol(w2)    # 1
neuron_df <- data.frame(
  layer = rep(c("Input", "Hidden", "Output"), c(n_input, n_hidden, n_output)),
  x = rep(c(1, 2, 3), c(n_input, n_hidden, n_output)),
  y = c(seq(1, n_input), seq(1, n_hidden), seq(1, n_output))
)
edges1 <- expand.grid(
  from = 1:n_input,
  to = (n_input + 1):(n_input + n_hidden)
)
edges1$weight <- as.vector(w1)
edges1$from_x <- 1
edges1$to_x <- 2
edges1$from_y <- neuron_df$y[edges1$from]
edges1$to_y <- neuron_df$y[edges1$to]
edges2 <- expand.grid(
  from = (n_input + 1):(n_input + n_hidden),
  to = n_input + n_hidden + 1
)
edges2$weight <- as.vector(w2)
edges2$from_x <- 2
edges2$to_x <- 3
edges2$from_y <- neuron_df$y[edges2$from]
edges2$to_y <- neuron_df$y[edges2$to]
edges_all <- rbind(edges1, edges2)
ggplot() +
  # Garis koneksi
  geom_segment(data = edges_all,
               aes(x = from_x, xend = to_x, y = from_y, yend = to_y,
                   color = weight, size = abs(weight))) +
  scale_color_gradient2(low = "blue", mid = "gray80", high = "red", midpoint = 0) +
  scale_size(range = c(0.2, 2)) +
  # Neuron titik
  geom_point(data = neuron_df,
             aes(x = x, y = y), size = 6, color = "black", fill = "white", shape = 21) +
  geom_text(data = neuron_df,
            aes(x = x, y = y, label = layer), vjust = -1.5, size = 4) +
  theme_void()

