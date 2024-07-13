'''
Imports
'''
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import roc_curve, auc
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import matplotlib.pyplot as plt

'''
Constants
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(epochs, model, learning_rate, train_dl, val_dl, min_epoch_train, patience, epsilon, log_file, model_save_path, criterion = nn.BCEWithLogitsLoss()):
    '''
    Training function for ISIC-2024 competition data
    Parameters:
        epochs: Number of epoches
        model: Model 
        learning_rate: model learning rate
        train_dl: Training data loader
        val_dl: Validation data loader
        min_epoch_train: Train for minimum epoches after which early-stopping kicks in
        patience: Patience for early-stopping
        epsilon: minimum required improvement in order to go on beyond early-stopping
        log_file: log file location to save logs
        model_save_path: location for saving trained model 
        criterion: Loss function, defaults to BCE with logit loss function
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Initialize CosineAnnealingLR scheduler
    multiplier = lambda epoch: 0.75
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=multiplier)      # multiply learning rate with 0.75 each epoch
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_val_pauc = -1.0  # Initialize with a very low value
    current_patience = 0  # Initialize patience counter

    with open(log_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Learning Rate', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score', 'Validation pAUC'])
        
        for epoch in range(epochs):
            print(f"\n | Epoch: {epoch+1}")
            total_loss = 0
            num_corr = 0
            num_samp = 0
            loop = tqdm(train_dl)
            model.train()

            for batch_idx, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels.float())
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                preds = torch.sigmoid(outputs)
                num_corr += ((preds > 0.5) == labels).sum()
                num_samp += preds.size(0)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(train_dl)
            acc = num_corr / num_samp
            print(f"| Epoch {epoch+1}/{epochs} total training loss: {total_loss}, average training loss: {avg_loss}.")
            
            print("On Validation Data:")
            
            model.eval()
            with torch.inference_mode():
                val_loss, val_acc, val_pre, val_rec, val_f1, val_pauc = evaluate(val_dl, model, criterion)
            print("learning rate:", scheduler.get_last_lr()[0])
            row = [epoch+1, scheduler.get_last_lr()[0], avg_loss, acc.item(), val_loss, val_acc, val_pre, val_rec, val_f1, val_pauc]
            csv_writer.writerow(row)

            if epoch + 1 > min_epoch_train:
                '''
                early-stopping code
                '''
                if val_pauc > best_val_pauc and (val_pauc - best_val_pauc) > epsilon:
                    best_val_pauc = val_pauc
                    print(f'Validation pAUC improved by more than {epsilon}, ({best_val_pauc} > {best_val_pauc})); saving model...')
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, model_save_path)
                    print(f'Model saved at {model_save_path}')
                    current_patience = 0  # Reset patience if there's an improvement
                else:
                    current_patience += 1
                    print(f'Validation pAUC did not improve. Patience left: {patience - current_patience}')
                    
                    if current_patience >= patience:
                        print(f'\n---Early stopping at epoch {epoch+1}.---')
                        break
            else:
                '''
                train for at least min_epoch_train epochs and keep saving best
                '''
                if val_pauc > best_val_pauc:
                    best_val_pauc = val_pauc
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, model_save_path)
                    print(f'Model saved at {model_save_path}')

            print(f'Current Best Validation pAUC: {best_val_pauc}')

            scheduler.step()  # Update learning rate for next epoch
        
    print('Training complete.')

    return best_val_pauc


def pauc_above_tpr(y_true, y_pred, min_tpr=0.80):
    '''
    Custom metric according to competition
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve
    https://www.kaggle.com/code/metric/isic-pauc-abovetpr
    '''
    y_true = abs(np.array(y_true) - 1)
    y_pred = -1.0 * np.array(y_pred)
    
    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("NaN values detected in inputs to pauc_above_tpr")
        return 0
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    max_fpr = 1 - min_tpr

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    
    if len(fpr) < 2:
        print("Warning: Not enough points to compute pAUC. Returning 0.")
        return 0
    
    partial_auc = auc(fpr, tpr)

    return partial_auc


def evaluate(loader, model, criterion = nn.BCEWithLogitsLoss()):
    '''
    Evaluate function for validation set
    '''
    metric = BinaryF1Score(threshold=0.5).to(DEVICE)
    prec = BinaryPrecision(threshold=0.5).to(DEVICE)
    recall = BinaryRecall(threshold=0.5).to(DEVICE)
    acc = BinaryAccuracy(threshold=0.5).to(DEVICE)
    loss = 0.0
    num_corr = 0
    num_samp = 0
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).squeeze(1)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print("NaN detected in model outputs")
                continue
            
            loss += criterion(outputs, labels.float()).item()
            preds = torch.sigmoid(outputs)
            num_corr += ((preds > 0.5) == labels).sum()
            num_samp += preds.size(0)
            metric.update(preds, labels)
            prec.update(preds, labels)
            recall.update(preds, labels)
            acc.update(preds, labels)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = loss / len(loader)
    accu = float(num_corr) / float(num_samp)
    pauc = pauc_above_tpr(all_labels, all_preds)
    
    print(f"Total loss: {loss}, Average loss: {avg_loss}")
    print(f"Got {num_corr}/{num_samp} correct with accuracy {accu*100:.2f}")
    print(f"pAUC above 80% TPR: {pauc:.3f}, Accuracy: {acc.compute().item():.3f}, precision: {prec.compute().item():.3f}, recall: {recall.compute().item():.3f}, F1Score: {metric.compute().item():.3f}")
    model.train()

    return avg_loss, acc.compute().item(), prec.compute().item(), recall.compute().item(), metric.compute().item(), pauc


def save_checkpoint(state, filename="my_checkpoint.pth"):
    '''
    To save model while training 
    Saves in working directory by default
    '''
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    '''
    
    '''
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def load_model(model_save_path = None, imagenet_weights_path = None):   # Use this for submission rather than load_checkpoint() defined above
    '''
    To load model during evaluation on test set
    '''
    if model_save_path:
        model = torchvision.models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=1)
        model.load_state_dict(torch.load(model_save_path)["state_dict"])
        model.to(DEVICE)
        model.eval()
    else:
        model = torchvision.models.resnet34(weights=None)
        model.load_state_dict(torch.load(imagenet_weights_path))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=1)
        model.to(DEVICE)
        
    return model


def create_submission(model, test_loader, submission_file_path):
    '''
    To predict class probabilities on test data and generate submission.csv file
    '''
    predictions = []
    image_ids = []

    with torch.no_grad():
        for inputs, image_names in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
            image_ids.extend(image_names)  # Append all image names from the batch

    # Check if the lengths match
    if len(image_ids) != len(predictions):
        print(f"Warning: Number of image IDs ({len(image_ids)}) does not match number of predictions ({len(predictions)})")

    # Create DataFrame
    submission_df = pd.DataFrame({
        'isic_id': image_ids,
        'target': predictions
    })

    # Save to CSV
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")


def visualize_train_images(images, titles=None):
    '''
    
    '''
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC and convert to numpy
        plt.imshow(image)
        if titles:
            plt.title(f"{titles[0][i]} (label: {titles[1][i]})")
        plt.axis('off')
    plt.show()
    """ Usage
    indices = np.random.choice(len(train_dataset_album), size=3, replace=False)
    images, label, image_ids  = zip(*[train_dataset_album[i] for i in indices])
    visualize_train_images(images, titles=[image_ids, label])
    """


def visualize_train_images(images, titles=None):
    '''
    
    '''
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC and convert to numpy
        plt.imshow(image)
        if titles:
            plt.title(f"{titles[0][i]} (label: {titles[1][i]})")
        plt.axis('off')
    plt.show()

    """ usage
    indices = np.random.choice(len(train_dataset), size=3, replace=False)
    images, label, image_ids  = zip(*[train_dataset[i] for i in indices])
    visualize_train_images(images, titles=[image_ids, label])
    """


def visualize_test_images(images, titles=None):
    '''
    
    '''
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC and convert to numpy
        plt.imshow(image)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()
    ''' Usage
    indices = np.random.choice(len(test_dataset), size=3, replace=False)
    images, image_ids  = zip(*[test_dataset[i] for i in indices])
    visualize_test_images(images, titles=image_ids)    
    '''


def plot_metrics_from_files(file_paths, save_path=None):
    '''
    
    '''
    num_files = len(file_paths)
    rows = 4  # Four rows for four types of plots
    cols = num_files  # One column per file

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10))  # Create a grid of subplots
    
    if num_files == 1:
        axes = axes[:, None]
    
    for i, file_path in enumerate(file_paths):
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract relevant columns
        epochs = df['Epoch']
        learning_rate = df['Learning Rate']
        train_loss = df['Training Loss']
        valid_loss = df['Validation Loss']
        valid_pAUC = df['Validation pAUC']
        valid_acc = df['Validation Accuracy']
        valid_prec = df['Validation Precision']
        valid_recall = df['Validation Recall']
        valid_f1score = df['Validation F1 Score']

        # Plot training loss and validation loss
        axes[0, i].plot(epochs, train_loss, label='Training Loss', marker='o')
        axes[0, i].plot(epochs, valid_loss, label='Validation Loss', marker='o')
        axes[0, i].set_title(f'File: {file_path.split("/")[-1]}')
        axes[0, i].set_xlabel('Epochs')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()

        # Plot epoch vs learning rate
        axes[1, i].plot(epochs, learning_rate, label='Learning Rate', marker='o', color='orange')
        axes[1, i].set_title(f'File: {file_path.split("/")[-1]}')
        axes[1, i].set_xlabel('Epochs')
        axes[1, i].set_ylabel('Learning Rate')
        axes[1, i].legend()

        # Plot validation pAUC
        axes[2, i].plot(epochs, valid_pAUC, label='Validation pAUC', marker='o', color='green')
        axes[2, i].set_title(f'File: {file_path.split("/")[-1]}')
        axes[2, i].set_xlabel('Epochs')
        axes[2, i].set_ylabel('Validation pAUC')
        axes[2, i].legend()

        # Plot accuracy, precision, recall, f1-score
        axes[3, i].plot(epochs, valid_acc, label = "Validation Accuracy", marker = 'o')
        axes[3, i].plot(epochs, valid_prec, label = "Validation Precision", marker = 'o')
        axes[3, i].plot(epochs, valid_recall, label = "Validation Recall", marker = 'o')
        axes[3, i].plot(epochs, valid_f1score, label = "Validation F1 Score", marker = 'o')
        axes[3, i].set_title(f'File: {file_path.split("/")[-1]}')
        axes[3, i].set_xlabel('Epochs')
        axes[3, i].set_ylabel('Validation Metrics')
        axes[3, i].legend()

    plt.tight_layout()  # Adjust the layout to prevent overlap

    if save_path:
        plt.savefig(save_path)  # Save the figure if a save path is provided

    # plt.show()