# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score

# Importing dataset as a csv file
file_path = r"C:\Users\Asus\OneDrive\Desktop\MSc Robotics\Data Modelling and Machine Intelligence\Course Work\Python\Qsar_data.csv"
qsar_data = pd.read_csv(file_path)

# Set up the matplotlib figure for a heatmap
plt.figure(figsize=(10, 8))

#Creating a correlation matrix which will be used in the heatmap
correlation_matrix = qsar_data.corr()

# Create a heatmap using Seaborn
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)

# Show the plot
plt.title('Heat Map for QSAR Dataset')
plt.show()

#______________________________________________________________________________________________________________________________________________

# Making a box plot

# Create a box plot for all columns
ax = qsar_data.boxplot(showfliers=True)

# Set x-axis labels as a range
num_columns = len(qsar_data.columns)
ax.set_xticklabels([str(i) for i in range(1, num_columns + 1)])

# Set y-axis limits to include negative values
plt.ylim(qsar_data.min().min(), qsar_data.max().max())

plt.title('Box Plot Plot for QSAR Dataset')
plt.xlabel('Column Number')
plt.ylabel('Feature Values')

# Display the plot
plt.show()

#_________________________________________________________________________________________________________________________________________________

# Removing duplicated code from the dataset

# Finding duplicate rows based on all columns
duplicates = qsar_data[qsar_data.duplicated(keep=False)]

# Create a new DataFrame without the duplicate rows
cleaned_qsar_data = qsar_data.drop_duplicates(keep=False)

# print(f'Dataset after removing duplicated datas: {cleaned_qsar_data}')

#__________________________________________________________________________________________________________________________________________________

# Using the new cleaned QSAR Dataset and standardizing it

# Separate numerical columns from the DataFrame
numerical_columns = cleaned_qsar_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
cleaned_qsar_data[numerical_columns[:-1]] = scaler.fit_transform(cleaned_qsar_data[numerical_columns[:-1]])

# print(f'Standardized Dataset: {cleaned_qsar_data}')

#___________________________________________________________________________________________________________________________________________________

# Performing the function of removing the outliers

# Calculate mean and standard deviation for each column
mean_values = cleaned_qsar_data.mean()
std_dev_values = cleaned_qsar_data.std()

# Defining a threshold for the z-score to identify outliers
threshold = 3 # We can take any number between 2 and 3 for this value. For this process we are considering it as 3.

# Calculate z-scores for each data point
z_scores = (cleaned_qsar_data - mean_values) / std_dev_values

# Filter the DataFrame to exclude rows with z-scores above the threshold
cleaned_qsar_data = cleaned_qsar_data[(abs(z_scores) < threshold).all(axis=1)]


# print(f'Dataset after removing outliers based on z-score value: {cleaned_qsar_data}')

#__________________________________________________________________________________________________________________________________________________

# PCA divided into two steps:
# 1. Plotting a explained variance ratio and determining the number of components for a given threshold
# 2. Perfoming PCA on the filtered_qsar_dataset to obtain a redcued dimensional datset


# Step 1:

# Separate features and target variable
X = cleaned_qsar_data.iloc[:, :-1]  # Exclude the last column as it is the target variable
y = cleaned_qsar_data.iloc[:, -1]   # Use the last column as the target variable

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Perform PCA
n_components = min(X_standardized.shape[0], X_standardized.shape[1])  # Using the smaller dimension for the number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_standardized)

# Create a DataFrame with the principal components
columns = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data=X_pca, columns=columns)

# Concatenate the principal components with the target variable
df_pca_with_target = pd.concat([df_pca, y], axis=1)

# Plot explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.75, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Each Principal Component')
plt.legend()
plt.show()

# Determine the number of components for a given threshold (e.g., 99% cumulative explained variance)
threshold = 0.99
n_components_threshold = len(cumulative_explained_variance[cumulative_explained_variance < threshold]) + 1

# print(f"Number of components for {threshold * 100}% cumulative explained variance: {n_components_threshold}")

#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

# ... (Step 2)

# Use the threshold calculated in Step 1 to determine the number of components
n_components_step2 = n_components_threshold
pca_step2 = PCA(n_components=n_components_step2)
X_pca_step2 = pca_step2.fit_transform(X_standardized)

# Create a DataFrame with the principal components
columns_step2 = [f'PC{i + 1}' for i in range(X_pca_step2.shape[1])]
qsar_data_pca_step2 = pd.DataFrame(data=X_pca_step2, columns=columns_step2)


# Concatenate the principal components with the target variable after resetting the index of y
qsar_data_after_pca = pd.concat([qsar_data_pca_step2, y.reset_index(drop=True)], axis=1)

print(f"The final dataset going to be used for machine learning algorithm: {qsar_data_after_pca}")

#______________________________________________________________________________________________________________________________________________________________________

# Doing Random Forrest Classification

# Extract features (X) and target variable (y)
X = qsar_data_after_pca.iloc[:, :-1].values  # All columns except the last one
y = qsar_data_after_pca.iloc[:, -1].values   # Last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
n_trees = 99
rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state=42)

# Train the model and track MSE during training
mse_values = []
for i in range(1, n_trees + 1):
    rf_classifier.n_estimators = i
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

    if(i == n_trees):
        print(f'Mean Squared Error: {mse}')

cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
disp.plot()
plt.show()

# Plot the Mean Squared Error during training
plt.plot(range(1, n_trees + 1), mse_values, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Random Forest Classifier Training Progress')
plt.show()

cv_scores = cross_val_score(rf_classifier, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')

#_________________________________________________________________________________________________________________________________________________________________

# Doing Random Forrest Regression

# Extract features (X) and target variable (y)
X = qsar_data_after_pca.iloc[:, :-1].values  # All columns except the last one
y = qsar_data_after_pca.iloc[:, -1].values   # Last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
n_trees = 99
rf_regressor = RandomForestRegressor(n_estimators=n_trees, random_state=42)

# Train the model and track MSE during training
mse_values = []
for i in range(1, n_trees + 1):
    rf_regressor.n_estimators = i
    rf_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_regressor.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

    if(i == n_trees):
        print(f'Mean Squared Error: {mse}')


# Plot the Mean Squared Error during training
plt.plot(range(1, n_trees + 1), mse_values, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Random Forest Regression Training Progress')
plt.show()

#Calculating the r-square value and printing it
r_squared = r2_score(y_test, y_pred)
print(f'R-squared Value: {r_squared}')

cv_scores = cross_val_score(rf_regressor, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')

#_____________________ THE END _____________________________________________
