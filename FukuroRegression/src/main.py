# Dependencies
import argparse
import torch
from config import DATA_PATH, MODEL_PATH, RANDOM_SEED, TRAIN_SIZE, LEARNING_RATE, DEVICE, DIAMETER
from utils import make_train_test, move_tensor_device, display_equation
from models import FukuroLinearRegression, FukuroParabolicRegression
from train import train_model
from test import test_model

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='input *.csv file name')
parser.add_argument('-t', '--type', type=str, choices=['linear', 'parabolic'], help='choosing model type')
parser.add_argument('-m', '--mode', type=str, choices=['training', 'testing'], help='mode options')
args = parser.parse_args()

torch.manual_seed(RANDOM_SEED)

if __name__ == '__main__':
    # Data processing
    FILE_PATH = DATA_PATH / args.file
    X_train, y_train, X_test, y_test = make_train_test(file_path=FILE_PATH, train_size=TRAIN_SIZE)

    # Define the model
    if args.type == 'linear':
        model = FukuroLinearRegression()
        MODEL_SAVE_PATH = MODEL_PATH / 'linear_model.pth'
    elif args.type == 'parabolic':
        model = FukuroParabolicRegression()
        MODEL_SAVE_PATH = MODEL_PATH / 'parabolic_model.pth'

    # Move everything to device
    model.to(DEVICE)
    X_train, y_train, X_test, y_test = move_tensor_device(X_train, y_train, X_test, y_test, target=DEVICE)

    # Mode
    if args.mode == 'training':
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
        best_model = train_model(X_train, y_train, X_test, y_test, 
                                                                model, loss_fn, optimizer, best_model_path=MODEL_SAVE_PATH)
        
        # display equation
        display_equation(best_model, args.type)
    else:
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        model.to(DEVICE)

        # display equation
        display_equation(model, args.type)

        # test and plot model 
        test_model(model, X_train, y_train, X_test, y_test, type=args.type)