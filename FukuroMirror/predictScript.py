import numpy as np
import pickle

# Predict
def predict_distance(model, pixel_x, pixel_y):
    """Predicts the distance for given pixel_x and pixel_y using GPR."""
    input_data = np.array([[pixel_x, pixel_y]])
    predicted_distance, _ = model.predict(input_data, return_std=True)
    return predicted_distance[0]

if __name__ == "__main__":
    # Load the model
    with open("models/gpr_model.pkl", "rb") as file:
        gpr_loaded = pickle.load(file)
    print("Model loaded successfully!")

    # Define testing variable
    test_x, test_y = 466, 192
    true_value = 176.77

    # Predict the distance
    test_pred = predict_distance(gpr_loaded, test_x, test_y)
    print(f"Predicted Distance: {test_pred:.3f} cm")
    print(f"Error: {abs(true_value - test_pred):.3f} cm")
