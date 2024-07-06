import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, FIG_SIZE

def make_train_test(file_path, train_size=0.8):
    """
        Preprocess the .csv data by performing train test split.
    """
    data = pd.read_csv(file_path)
    try:
        X = data['X'].values
    except:
        X = data['x'].values
    y = data['y'].values

    X_tensor = torch.Tensor(X).type(torch.float).view(-1, 1)
    y_tensor = torch.Tensor(y).type(torch.float).view(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, train_size=train_size, random_state=RANDOM_SEED)

    return X_train, y_train, X_test, y_test


def move_tensor_device(X_train, y_train, X_test, y_test, y_preds=None, target=None):
    """
        Move tensor to target device.
    """
    if target == 'cuda':
        X_train = X_train.to(target)
        y_train = y_train.to(target)
        X_test = X_test.to(target)
        y_test = y_test.to(target)
    else:
        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_test = X_test.cpu()
        y_test = y_test.cpu()
        if y_preds is not None: 
            y_preds = y_preds.cpu()
            return X_train, y_train, X_test, y_test, y_preds
        
    return X_train, y_train, X_test, y_test
    
def plot_data(X_train, y_train, X_test, y_test, y_preds=None):
    """
        Plot input data. If prediction exist also plot the predictions.
    """
    plt.figure(figsize=FIG_SIZE)
    # plot training data
    plt.scatter(X_train, y_train, color='b', s=10, label='Train Data')
    # plot test data
    plt.scatter(X_test, y_test, color='g', s=10, label='Test Data') 
    # plot prediction
    if y_preds is not None:
        plt.scatter(X_test, y_preds, color='r', s=10, label='Predictions')

    plt.title("2D Input Data Visualization")
    plt.legend()
    plt.show()

def plot_3d_parabolic(a, diameter):
    """
        3D plotting after perforiming parabolic regression successfully.
        It works by only using the 'a' from equation: ax^2+bx+c for plotting
        in the center of 3D plane. 
    """
    # Define the range for r and theta
    r = np.linspace(0, diameter, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Create a meshgrid for r and theta
    R, Theta = np.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = a * R**2

    # Plotting
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.title("3D Visualization of Parabolic Equation")
    plt.show()

def display_equation(model: torch.nn.Module, type=None):
    """
        Display the estimated function.
    """
    params = tuple(model.state_dict().items())
    constant = [c[1].item() for c in params]
    
    print('='*100)
    if type == 'linear':
        print(f"{type} equation is : y = {constant[0]} x + {constant[1]}")
    else:
        print(f"{type} equation is : y = {constant[0]} x^2 + {constant[1]} x + {constant[2]}")
    print('='*100)