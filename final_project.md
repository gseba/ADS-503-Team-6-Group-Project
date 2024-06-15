# Kidney Stone Prediction based on Urine Analysis

```
Carlos Fuentes Rosa
Ghassan Seba
Muris Saab

Universtity of San Diego
ADS-503 Final Project
Spring 2024
```


### 1. Problem Statement and Justification

To predict the presence of kidney stones based on urine analysis data, focusing on six physical characteristics of urine.

Understanding the relationship between urine characteristics and kidney stone formation can help in early detection and prevention, leading to better patient outcomes.

### 2. Exploratory Data Analysis (EDA)

-   **Graphical Analysis:** Visualize the distribution of each feature and its relationship with the target variable.
-   **Non-Graphical Analysis:** Summary statistics and correlation matrix.

### 3. Data Wrangling and Pre-processing

-   **Handling Missing Values:** Check and handle any missing data.
-   **Outliers:** Identify and address outliers.
-   **Correlated Features:** Identify and address multicollinearity.

### 4. Data Splitting

-   Split the data into training, validation, and test sets to ensure robust model evaluation.

### 5. Model Strategies

-   **Research Questions:** Which physical characteristics are most predictive of kidney stones?
-   **Analytics Methods:** Explore logistic regression, decision trees, random forests, and other relevant models.

### 6. Validation and Testing

-   **Model Tuning:** Hyperparameter tuning using cross-validation.
-   **Evaluation:** Assess model performance using accuracy, precision, recall, and AUC-ROC.

### 7. Results and Final Model Selection

-   Present the final model with performance metrics.
-   Discuss the importance of each feature in the prediction.

### 8. Discussion and Conclusions

-   Address the initial problem statement.
-   Provide insights and potential next steps for further research or implementation.

### 9. Appendix

-   Include reproducible R code used for analysis and modeling.

### Technical Report


### Executive Summary PowerPoint

-   **Slide 1:** Introduction and Problem Statement
-   **Slide 2:** Data and Methodology
-   **Slide 3:** EDA and Key Findings
-   **Slide 4:** Model Results and Performance
-   **Slide 5:** Conclusions and Recommendations

### Video Presentation

-   Create a narrated presentation explaining the project's objectives, methodology, results, and conclusions.
-   Ensure good sound quality and equal participation from all team members.


### Summary Statistics and Correlation Analysis

**Summary Statistics:** The summary statistics provide an overview of the central tendency, dispersion, and shape of the dataset's distribution.

**Correlation Matrix:** The correlation matrix shows the linear relationships between pairs of variables. Key observations include: - **Gravity** and **Osmolarity** have a strong positive correlation (0.86). - **Gravity** and **Urea** also show a strong positive correlation (0.82). - **Calcium** and **Target** have a moderate positive correlation (0.54), indicating that higher calcium concentration may be associated with the presence of kidney stones.

# Appendix


### 1. Load Libraries and Data

First, ensure you have the necessary libraries installed and then load the data:

```{r}
# Load libraries
library(tidyverse)
library(caret)
library(GGally)
library(corrplot)

# Load the data
data <- read.csv("kindey_stone_urine_analysis.csv")

# View the first few rows
head(data)
```

### 2. Data Wrangling and Pre-processing

#### Handling Missing Values

Check and handle missing values:

```{r}
# Check for missing values
sum(is.na(data))

# If any missing values, handle them (e.g., by removing rows with missing values)
data <- na.omit(data)
```

#### Outliers

Identify and handle outliers:

```{r}
# Boxplot to identify outliers
boxplot(data)

# Handle outliers (example: removing outliers based on z-score)
data <- data %>%
  filter(abs(scale(gravity)) < 3,
         abs(scale(ph)) < 3,
         abs(scale(osmo)) < 3,
         abs(scale(cond)) < 3,
         abs(scale(urea)) < 3,
         abs(scale(calc)) < 3)
```

#### Correlated Features

Check for multicollinearity:

```{r}
# Correlation matrix
cor_matrix <- cor(data[,-ncol(data)])
corrplot(cor_matrix, method="circle")

# If highly correlated features exist, you can consider removing or combining them
```

### 3. Data Splitting

Split the data into training and test sets:

```{r}
# Set seed for reproducibility
set.seed(123)

# Split the data
train_index <- createDataPartition(data$target, p=0.8, list=FALSE)
train_data <- data[train_index,]
test_data <- data[-train_index,]
```

### 4. Model Strategies and Training

Train different models:

```{r}
# Logistic Regression
log_model <- train(target ~ ., data=train_data, method="glm", family="binomial")

# Random Forest
rf_model <- train(target ~ ., data=train_data, method="rf")

# Evaluate models
log_preds <- predict(log_model, newdata=test_data)
rf_preds <- predict(rf_model, newdata=test_data)

# Confusion Matrix
confusionMatrix(log_preds, test_data$target)
confusionMatrix(rf_preds, test_data$target)
```

### 5. Model Validation and Selection

Use cross-validation for model tuning:

```{r}
# Cross-validation for logistic regression
log_model_cv <- train(target ~ ., data=train_data, method="glm", family="binomial",
                      trControl=trainControl(method="cv", number=10))

# Cross-validation for random forest
rf_model_cv <- train(target ~ ., data=train_data, method="rf",
                     trControl=trainControl(method="cv", number=10))

# Compare models
resamples(list(Logistic=log_model_cv, RandomForest=rf_model_cv)) %>%
  summary()
```

### 6. Results and Final Model Selection

Select the best model based on performance:

```{r}
# Evaluate the best model on the test set
best_model <- rf_model_cv  # assuming random forest performed better
final_preds <- predict(best_model, newdata=test_data)

# Confusion Matrix
confusionMatrix(final_preds, test_data$target)

# ROC Curve
roc_curve <- roc(test_data$target, as.numeric(final_preds))
plot(roc_curve)
```




