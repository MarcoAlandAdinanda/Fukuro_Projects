"""
    This is a dummy data maker to perform sanity check of the models.
    By using this formula (e.g.) : y = (0.1)X^2 + (2)X + 1
    a = 0.1
    b = 2
    c = 1
"""
import torch
import argparse
import pandas as pd
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', type=str, choices=['linear', 'parabolic'], help='choosing model type')
args = parser.parse_args()

if args.type == 'linear':
    # PARAMETERS
    a = 3
    b = 5

    # Make data
    X = torch.arange(-10, 10, 1)#.view(-1, 1)
    y = a * X + b

    filename = 'linear_dummy_data.csv'

elif args.type == 'parabolic':
    # PARAMETERS
    a = 0.1
    b = 2
    c = 1   

    # Make data
    X = torch.arange(-20, 0, 1)#.view(-1, 1)
    y = a * (X**2) + b*X + c

    filename = 'parabolic_dummy_data.csv'

# Check the values
if __name__ == '__main__':
    data = {'X': X.tolist(),
            'y': y.tolist()}
    df = pd.DataFrame(data)
    PATH = DATA_PATH / filename
    df.to_csv(PATH)
    print(f"Dummy data saved at '{PATH}'")