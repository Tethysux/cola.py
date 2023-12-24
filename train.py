import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
your_dataframe = pd.read_csv("coca.csv")

# 1. Handling Missing Values:
# Drop rows with missing values
your_dataframe.dropna(inplace=True)

# 2. Dealing with Outliers in Numerical Columns (Open, High, Low, Close, Volume):
# Identify outliers using z-scores
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
z_scores = np.abs(stats.zscore(your_dataframe[numeric_columns]))
your_dataframe_no_outliers = your_dataframe[(z_scores < 3).all(axis=1)]

# 3. Cleaning Feature Names (Replace spaces with underscores):
your_dataframe.columns = your_dataframe.columns.str.strip().str.replace(' ', '_')

# 4. Reindexing:
# After modifying your DataFrame, reset the index
your_dataframe.reset_index(drop=True, inplace=True)

# 5. Split the data into features and target
features = your_dataframe_no_outliers[['Open', 'High', 'Low', 'Volume']]
target = your_dataframe_no_outliers['Close']

# 6. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 7. Create Ridge regression model
ridge_model = Ridge(alpha=1.0)

# 8. Create a pipeline with a StandardScaler and the Ridge regression model
ridge_pipeline = make_pipeline(StandardScaler(), ridge_model)

# 9. Create a hybrid model using TransformedTargetRegressor with Ridge regression
hybrid_model = TransformedTargetRegressor(regressor=ridge_pipeline, transformer=None)

# 10. Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# 11. Transform both the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. Train the hybrid model on the scaled training data
hybrid_model.fit(X_train_scaled, y_train)

# 13. Make predictions on the scaled test set
hybrid_predictions = hybrid_model.predict(X_test_scaled)

# 14. Evaluate the hybrid model
hybrid_mse = mean_squared_error(y_test, hybrid_predictions)
print(f'Hybrid Model Mean Squared Error: {hybrid_mse}')

# 15. Use the trained hybrid model for predictions
predictions = hybrid_model.predict(X_test_scaled)

# 'predictions' now contains the predicted values for your target variable
# You can use these predictions for evaluation or any other further analysis
