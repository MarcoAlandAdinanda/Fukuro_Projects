import torch
import matplotlib.pyplot as plt
from config import FIG_SIZE, EPOCHS, PATIENCE

def plot_training_loss(train_loss_log, test_loss_log):
    """
        Plot training and test loss to evaluate the training process
    """
    plt.figure(figsize=FIG_SIZE)
    # Plot training loss
    plt.plot(train_loss_log, color='r', label='Train Loss')
    # Plot test loss
    plt.plot(test_loss_log, color='g', label='Test Loss')

    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()

def train_model(X_train, y_train, X_test, y_test, model, loss_fn, optimizer, 
                  epochs=EPOCHS, patience=PATIENCE, best_model_path=None):
    """
        Training loop function to train the model.
    """
    train_loss_log = []
    test_loss_log = []
    past_test_loss = 999999999
    n_patience = 0

    for epoch in range(epochs):
        ### TRAIN ###
        # Calculating train loss
        model.train()
        y_preds_train = model(X_train)
        train_loss = loss_fn(y_preds_train, y_train)
        # Learning
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        ### Test ###
        # Calculating test loss
        model.eval()
        with torch.inference_mode():
            y_preds_test = model(X_test)
            test_loss = loss_fn(y_preds_test, y_test)
        
        # training log
        train_loss_log.append(train_loss.item())
        test_loss_log.append(test_loss.item())

        if epoch % 100 == 0:
            # Checkpoint
            if test_loss < past_test_loss:
                past_test_loss = test_loss
                n_patience = 0
                torch.save(obj=model.state_dict(), f=best_model_path)
            else:
                n_patience += 1
                if n_patience > patience:
                    print("------EarlyStopping Triggred------")
                    break
            print(f'Epoch: {epoch} | Training Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Patience: {n_patience}')
    
    # plot training and test loss
    plot_training_loss(train_loss_log, test_loss_log)

    return model