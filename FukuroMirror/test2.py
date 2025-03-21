# Import required libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('dataset/pixel_field_omni.csv', sep=';')  # Update with your file path
X = data[['pixel_x', 'pixel_y']].values
y = data[['field_x', 'field_y']].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Create and train SVR model
svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
model = MultiOutputRegressor(svr)
model.fit(X_train_scaled, y_train_scaled)

# Make predictions
train_pred_scaled = model.predict(X_train_scaled)
test_pred_scaled = model.predict(X_test_scaled)

# Inverse transform predictions
train_pred = scaler_y.inverse_transform(train_pred_scaled)
test_pred = scaler_y.inverse_transform(test_pred_scaled)

# Calculate metrics
def print_metrics(y_true, y_pred, label):
    print(f"\n{label} Metrics:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

print_metrics(y_train, train_pred, "Training")
print_metrics(y_test, test_pred, "Testing")

# Visualization
plt.figure(figsize=(12, 6))

# Plot field_x predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], test_pred[:, 0], alpha=0.5)
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])], 
         [min(y_test[:, 0]), max(y_test[:, 0])], 'r--')
plt.xlabel('True field_x')
plt.ylabel('Predicted field_x')
plt.title('Field X Coordinate Prediction')

# Plot field_y predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], test_pred[:, 1], alpha=0.5)
plt.plot([min(y_test[:, 1]), max(y_test[:, 1])], 
         [min(y_test[:, 1]), max(y_test[:, 1])], 'r--')
plt.xlabel('True field_y')
plt.ylabel('Predicted field_y')
plt.title('Field Y Coordinate Prediction')

plt.tight_layout()
plt.show()

# Print sample predictions
print("\nSample Predictions:")
print(f"{'Input (pixel)':<25} {'True Output (field)':<25} {'Predicted Output (field)':<25}")
for i in range(5):
    print(f"{str(X_test[i]):<25} {str(y_test[i]):<25} {str(test_pred[i].round(2)):<25}")


# Add menu for manual testing
print("\nManual Testing Interface:")
while True:
    print("\nOptions:")
    print("1. Test a single pixel coordinate")
    print("2. Exit program")
    choice = input("Please enter your choice (1/2): ").strip()
    
    if choice == '1':
        try:
            # Get user input
            pixel_x = float(input("Enter pixel_x coordinate: "))
            pixel_y = float(input("Enter pixel_y coordinate: "))
            user_sample = np.array([[pixel_x, pixel_y]])
            
            # Scale input
            scaled_sample = scaler_X.transform(user_sample)
            
            # Make prediction
            scaled_pred = model.predict(scaled_sample)
            
            # Inverse transform prediction
            field_pred = scaler_y.inverse_transform(scaled_pred)
            
            # Display results
            print(f"\nPredicted Field Coordinates:")
            print(f"field_x: {field_pred[0][0]:.4f}")
            print(f"field_y: {field_pred[0][1]:.4f}")
            
        except ValueError:
            print("Invalid input! Please enter numerical values.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    elif choice == '2':
        print("Exiting program...")
        break
    
    else:
        print("Invalid option! Please choose 1 or 2.")