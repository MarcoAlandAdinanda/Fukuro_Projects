import torch
from config import DIAMETER
from utils import move_tensor_device, plot_data, plot_3d_parabolic

def test_model(model, X_train, y_train, X_test, y_test, type=None):
    """
        Test the model by performing plot_data and plot 3d
    """
    model.eval()
    with torch.inference_mode():
        y_preds = model(X_test)
    X_train, y_train, X_test, y_test, y_preds = move_tensor_device(X_train, y_train, X_test, y_test, y_preds)
    
    # plot the prediction
    plot_data(X_train, y_train, X_test, y_test, y_preds)
    
    # plot 3d
    if type == 'parabolic':
        plot_3d_parabolic(model.cpu().state_dict()['a'], diameter=DIAMETER)