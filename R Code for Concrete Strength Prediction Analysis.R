# -----------------------------------------------------------------------------
# A Comparative Statistical Analysis of Machine Learning Models for
# Predicting Concrete Compressive Strength
#
# R Script for Data Analysis, Modeling, and Visualization
# Author: Dr. Debashis Chatterjee 
# Date: August 21, 2025
# -----------------------------------------------------------------------------


# --- 1. SETUP: LOAD PACKAGES AND CREATE DIRECTORY ---

# This section ensures all necessary packages are installed and loaded.
# We use the 'pacman' package to streamline this process.
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,    # For data manipulation and visualization (ggplot2)
  readxl,       # To read Excel files
  corrplot,     # For beautiful correlation plots
  mgcv,         # For Generalized Additive Models (GAM)
  e1071,        # For Support Vector Machines (SVM)
  randomForest, # For Random Forest models
  gt,           # For creating beautiful tables
  caret,        # For data splitting and model training utilities
  here          # For reproducible file paths
)

# Create a directory to save all our outputs (plots and tables)
# The 'here' package makes sure the path is correct regardless of OS.
output_dir <- here::here("analysis_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("Created directory to save results at:", output_dir, "\n")
} else {
  cat("Output directory already exists at:", output_dir, "\n")
}


# --- 2. DATA LOADING AND PREPARATION ---

# The dataset is hosted at the UCI Machine Learning Repository.
# We will download it directly from the URL.
cat("Downloading and loading the dataset...\n")
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
download.file(url, destfile = "Concrete_Data.xls", mode = "wb")
concrete_df <- read_excel("Concrete_Data.xls")

# The original column names have spaces and units, which can be tricky in R.
# Let's rename them to be more programmatic and easier to use.
original_names <- colnames(concrete_df)
new_names <- c(
  "cement", "blast_furnace_slag", "fly_ash", "water",
  "superplasticizer", "coarse_aggregate", "fine_aggregate",
  "age", "compressive_strength"
)

# Assign the new names and print a mapping for clarity
names(concrete_df) <- new_names
name_mapping <- data.frame(Original = original_names, New = new_names)
print("Column names have been updated for easier use:")
print(name_mapping)

# Display the first few rows to confirm it's loaded correctly
cat("\nFirst 6 rows of the dataset:\n")
print(head(concrete_df))


# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---

cat("\nStarting Exploratory Data Analysis (EDA)...\n")

# 3.1. Summary Statistics Table
# Create a descriptive statistics table and format it nicely with 'gt'.
summary_stats_table <- concrete_df %>%
  summary() %>%
  as.data.frame() %>%
  select(-Var1) %>%
  rename(Statistic = Freq) %>%
  mutate(Statistic = str_extract(Statistic, "\\w+")) %>%
  gt() %>%
  tab_header(
    title = "Table 1: Descriptive Statistics of Concrete Components",
    subtitle = "Summary of all variables in the dataset (N=1030)"
  ) %>%
  fmt_number(columns = everything(), decimals = 2)

# Print and save the table
print(summary_stats_table)
gtsave(summary_stats_table, filename = file.path(output_dir, "table1_summary_statistics.html"))
cat("Saved summary statistics table.\n")


# 3.2. Distribution of Variables Plot
# Visualize the distribution of each variable using histograms.
distributions_plot <- concrete_df %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_density(color = "firebrick", size = 1) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  labs(
    title = "Figure 1: Distribution of All Variables",
    subtitle = "Histograms and density plots for each component and strength",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal(base_size = 14) +
  theme(strip.text = element_text(face = "bold"))

# Print and save the plot
print(distributions_plot)
ggsave(
  file.path(output_dir, "figure1_variable_distributions.png"),
  plot = distributions_plot, width = 12, height = 10, dpi = 300
)
cat("Saved variable distributions plot.\n")


# 3.3. Correlation Matrix Plot
# Calculate the correlation matrix to understand linear relationships.
cor_matrix <- cor(concrete_df)

# Create a colorful and informative correlation plot
png(file.path(output_dir, "figure2_correlation_matrix.png"), width = 1000, height = 1000, res = 150)
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         order = "hclust",
         addCoef.col = "black", # Add correlation coefficients
         tl.col = "black", tl.srt = 45, # Text label color and rotation
         diag = FALSE, # Hide correlation of variables with themselves
         title = "\n\nFigure 2: Correlation Matrix of Variables",
         mar=c(0,0,2,0) # Adjust margins for title
)
dev.off()
cat("Saved correlation matrix plot.\n")
# Also print the matrix to the console
cat("\nCorrelation Matrix:\n")
print(round(cor_matrix, 2))


# --- 4. DATA SPLITTING: TRAINING AND TESTING SETS ---

cat("\nSplitting data into training (80%) and testing (20%) sets...\n")
set.seed(123) # for reproducibility
train_indices <- createDataPartition(concrete_df$compressive_strength, p = 0.8, list = FALSE)
train_data <- concrete_df[train_indices, ]
test_data <- concrete_df[-train_indices, ]

cat("Training set dimensions:", dim(train_data), "\n")
cat("Testing set dimensions:", dim(test_data), "\n")


# --- 5. MODEL BUILDING AND EVALUATION ---

cat("\nBuilding and evaluating models...\n")

# We will store the results of each model in this list
model_results <- list()

# Define a function to calculate metrics (RMSE and R-squared)
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  r_squared <- 1 - (sum((actual - predicted)^2) / sum((actual - mean(actual))^2))
  return(list(RMSE = rmse, R_squared = r_squared))
}

# 5.1. Model 1: Multiple Linear Regression (MLR)
cat("Training Multiple Linear Regression (MLR)...\n")
mlr_model <- lm(compressive_strength ~ ., data = train_data)
mlr_predictions <- predict(mlr_model, newdata = test_data)
mlr_metrics <- calculate_metrics(test_data$compressive_strength, mlr_predictions)
model_results$MLR <- mlr_metrics
print(summary(mlr_model))
cat("MLR Performance:", "RMSE =", round(mlr_metrics$RMSE, 3), "| R-squared =", round(mlr_metrics$R_squared, 3), "\n")

# 5.2. Model 2: Generalized Additive Model (GAM)
cat("\nTraining Generalized Additive Model (GAM)...\n")
# s() indicates a smooth term for each predictor
gam_model <- gam(compressive_strength ~ s(cement) + s(blast_furnace_slag) + s(fly_ash) +
                   s(water) + s(superplasticizer) + s(coarse_aggregate) +
                   s(fine_aggregate) + s(age), data = train_data)
gam_predictions <- predict(gam_model, newdata = test_data)
gam_metrics <- calculate_metrics(test_data$compressive_strength, gam_predictions)
model_results$GAM <- gam_metrics
cat("GAM Performance:", "RMSE =", round(gam_metrics$RMSE, 3), "| R-squared =", round(gam_metrics$R_squared, 3), "\n")

# Plot the smooth components of the GAM model
gam_plots <- plot(gam_model, pages = 1, all.terms = TRUE, rug = TRUE, se = TRUE, shade = TRUE)
png(file.path(output_dir, "figure3_gam_smooth_functions.png"), width = 1200, height = 1000, res = 150)
plot(gam_model, pages = 1, all.terms = TRUE, rug = TRUE, se = TRUE, shade = TRUE,
     main = "Figure 3: Smooth Functions from GAM")
dev.off()
cat("Saved GAM smooth functions plot.\n")


# 5.3. Model 3: Support Vector Machine (SVM)
cat("\nTraining Support Vector Machine (SVM)...\n")
# We use a radial basis function kernel. Tuning might improve results further.
svm_model <- svm(compressive_strength ~ ., data = train_data, kernel = "radial")
svm_predictions <- predict(svm_model, newdata = test_data)
svm_metrics <- calculate_metrics(test_data$compressive_strength, svm_predictions)
model_results$SVM <- svm_metrics
cat("SVM Performance:", "RMSE =", round(svm_metrics$RMSE, 3), "| R-squared =", round(svm_metrics$R_squared, 3), "\n")


# 5.4. Model 4: Random Forest (RF)
cat("\nTraining Random Forest (RF)...\n")
# ntree=500 is a common default. importance=TRUE allows us to inspect variable importance.
set.seed(123) # for reproducibility of the RF model
rf_model <- randomForest(compressive_strength ~ ., data = train_data, ntree = 500, importance = TRUE)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_metrics <- calculate_metrics(test_data$compressive_strength, rf_predictions)
model_results$RF <- rf_metrics
cat("RF Performance:", "RMSE =", round(rf_metrics$RMSE, 3), "| R-squared =", round(rf_metrics$R_squared, 3), "\n")

# Plot variable importance from the Random Forest model
importance_data <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column(var = "Variable") %>%
  arrange(desc(`%IncMSE`)) %>%
  mutate(Variable = fct_inorder(Variable))

var_imp_plot <- ggplot(importance_data, aes(x = `%IncMSE`, y = Variable)) +
  geom_col(fill = "forestgreen", alpha = 0.8) +
  labs(
    title = "Figure 4: Random Forest Variable Importance",
    subtitle = "Based on the increase in Mean Squared Error (%IncMSE)",
    x = "Increase in Mean Squared Error (%) when Variable is Permuted",
    y = "Variable"
  ) +
  theme_minimal(base_size = 14)

print(var_imp_plot)
ggsave(
  file.path(output_dir, "figure4_rf_variable_importance.png"),
  plot = var_imp_plot, width = 10, height = 8, dpi = 300
)
cat("Saved Random Forest variable importance plot.\n")


# --- 6. FINAL RESULTS COMPARISON ---

cat("\nGenerating final comparison of all models...\n")

# Combine all results into a single data frame
comparison_df <- do.call(rbind, lapply(model_results, as.data.frame)) %>%
  rownames_to_column(var = "Model") %>%
  rename(`RMSE (MPa)` = RMSE, `R-squared` = R_squared) %>%
  arrange(desc(`R-squared`))

# Create a final comparison table using 'gt'
final_table <- comparison_df %>%
  gt() %>%
  tab_header(
    title = "Table 2: Performance Comparison of Predictive Models",
    subtitle = "Metrics evaluated on the held-out test set (20% of data)"
  ) %>%
  fmt_number(columns = c("RMSE (MPa)", "R-squared"), decimals = 3) %>%
  cols_align(align = "center", columns = everything())

# Print and save the final table
print(final_table)
gtsave(final_table, filename = file.path(output_dir, "table2_model_comparison.html"))
cat("Saved final model comparison table.\n")

# Create a final comparison plot
comparison_plot_data <- comparison_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

final_plot <- ggplot(comparison_plot_data, aes(x = fct_reorder(Model, Value), y = Value, fill = Model)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  coord_flip() +
  labs(
    title = "Figure 5: Model Performance Comparison",
    subtitle = "Lower RMSE and Higher R-squared are better",
    x = "Model",
    y = "Metric Value"
  ) +
  theme_minimal(base_size = 14) +
  theme(strip.text = element_text(face = "bold"))

print(final_plot)
ggsave(
  file.path(output_dir, "figure5_model_comparison_plot.png"),
  plot = final_plot, width = 10, height = 6, dpi = 300
)
cat("Saved final model comparison plot.\n")

cat("\n--- ANALYSIS COMPLETE ---\n")
cat("All outputs have been saved to the 'analysis_results' folder.\n")

