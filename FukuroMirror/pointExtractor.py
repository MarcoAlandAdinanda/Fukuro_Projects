import cv2
import numpy as np
import csv

def get_click_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param["hsv"]
        color = hsv[y, x]  # Get HSV color at clicked point
        param["selected_color"] = color

def select_point_order(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["selected_points"].append((x, y))
        print(f"Point selected: ({x}, {y})")

def highlight_selected_color(image_path, output_csv='highlighted_points.csv'):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a window to select color
    cv2.namedWindow('Select Color')
    param = {"hsv": hsv, "selected_color": None}
    cv2.setMouseCallback('Select Color', get_click_color, param)
    
    while True:
        cv2.imshow('Select Color', image)
        if cv2.waitKey(1) & 0xFF == ord('q') or param["selected_color"] is not None:
            break
    
    cv2.destroyWindow('Select Color')
    
    if param["selected_color"] is None:
        print("No color selected.")
        return
    
    selected_h, selected_s, selected_v = param["selected_color"]
    
    # Define lower and upper bounds with an error range
    error = np.array([10, 50, 50])
    lower_bound = np.maximum([0, 0, 0], np.array([selected_h, selected_s, selected_v]) - error)
    upper_bound = np.minimum([179, 255, 255], np.array([selected_h, selected_s, selected_v]) + error)
    
    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    
    # Create a window to select point order
    cv2.namedWindow('Select Order')
    param = {"selected_points": []}
    cv2.setMouseCallback('Select Order', select_point_order, param)
    
    while True:
        img_copy = image.copy()
        for idx, (px, py) in enumerate(param["selected_points"]):
            cv2.putText(img_copy, str(idx + 1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('Select Order', img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(param["selected_points"]) == len(points):
            break
    
    cv2.destroyWindow('Select Order')
    
    # Save coordinates to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "X", "Y"])
        for i, (px, py) in enumerate(param["selected_points"], start=1):
            writer.writerow([i, px, py])
    
    # Apply the mask to highlight the selected color
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Show results
    cv2.imshow('Original Image', image)
    cv2.imshow('Highlighted Color', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'dataset/real_photo.png'  
highlight_selected_color(image_path, output_csv="real_test.csv")
