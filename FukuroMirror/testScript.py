import joblib
import numpy as np

regressor_x = joblib.load('regressor_x.pkl')
regressor_y = joblib.load('regressor_y.pkl')
poly_transform = joblib.load('poly_transform.pkl')

def predict_field_coordinates(pixel_x, pixel_y):
    input_data = np.array([[pixel_x, pixel_y]])
    input_poly = poly_transform.transform(input_data)
    
    field_x = regressor_x.predict(input_poly)
    field_y = regressor_y.predict(input_poly)
    
    return field_x[0], field_y[0]

# Example
if __name__ == "__main__":
    pixel_x, pixel_y = 400, 222  # Replace 
    field_x, field_y = predict_field_coordinates(pixel_x, pixel_y)
    print(f"Predicted Field Coordinates: ({field_x}, {field_y})")
