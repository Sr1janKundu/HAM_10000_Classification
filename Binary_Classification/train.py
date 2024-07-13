'''
Imports
'''
import torch, torchvision
import torch.nn as nn
import utils
import dataset_dataloader



'''
Constants
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
LEARNING_RATE = 0.0001
MIN_EPOCH_TRAIN = 15
PATIENCE = 5
EPSILON = 0.0005


'''
Paths
'''
ROOT_DATA_PATH = 'E:\\HAM_10000\Data\\HAM10000_images_all'
METADATA_PATH = 'E:\\HAM_10000\\HAM_10000_Classification\\Binary_Classification\\HAM10000_metadata_binary.csv'
MODEL_SAVE_PATH = 'E:\\HAM_10000\\HAM_10000_Classification\\Binary_Classification\\ResNet34_HAM10000_1-1.pth'
LOG_FILE = 'E:\\HAM_10000\\HAM_10000_Classification\\Binary_Classification\\log.csv'
METRICS_PLOT_PNG = 'E:\\HAM_10000\\HAM_10000_Classification\\Binary_Classification\\metrics.png'


def lesgoo():
    model_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
    model_resnet.to(DEVICE)
    train_dl, val_dl, test_dl = dataset_dataloader.get_loaders(root_data_path=ROOT_DATA_PATH, metadata_path=METADATA_PATH)
    _ = utils.train(epochs=EPOCHS, 
                    model = model_resnet, 
                    learning_rate = LEARNING_RATE, 
                    train_dl = train_dl, 
                    val_dl = val_dl, 
                    min_epoch_train = MIN_EPOCH_TRAIN, 
                    patience = PATIENCE, 
                    epsilon = EPSILON, 
                    log_file = LOG_FILE, 
                    model_save_path=MODEL_SAVE_PATH)
    utils.plot_metrics_from_files([LOG_FILE], METRICS_PLOT_PNG)
    print("\n\nOn Test Data:")
    trained_model = utils.load_model(model_save_path = MODEL_SAVE_PATH)
    trained_model.eval()
    utils.evaluate(test_dl, trained_model)

if __name__ == '__main__':
    lesgoo()