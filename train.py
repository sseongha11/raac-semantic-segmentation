# USAGE EXAMPLES:
# python train.py --models unet --output /content/drive/MyDrive/raac/output/unet/
# python train.py --models unet++ --output /content/drive/MyDrive/raac/output/unetpp
# python train.py --models deeplabv3+ --output /content/drive/MyDrive/raac/output/deeplabv3p/
# python train.py --models fpn --output /content/drive/MyDrive/raac/output/fpn/
# python train.py --models manet --output /content/drive/MyDrive/raac/output/manet/
# python train.py --models linknet --output /content/drive/MyDrive/raac/output/linknet/
# python train.py --models pspnet --output /content/drive/MyDrive/raac/output/pspnet/
# python train.py --models pan --output /content/drive/MyDrive/raac/output/pan/

import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import argparse
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as su
from torch.utils.data import DataLoader
from tqdm import tqdm
from raac import config
from raac.dataset import CracksDataset
from raac.helper_functions import get_validation_augmentation, get_preprocessing, get_training_augmentation
from raac import TransUNet


def plot_metric(train_df, valid_df, metric, ylabel, title, output_path):
    plt.figure(figsize=(20, 8))
    plt.plot(train_df.index.tolist(), train_df[metric].tolist(), lw=3, label='Train')
    plt.plot(valid_df.index.tolist(), valid_df[metric].tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(output_path)
    plt.show()


def main(args):
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    MODELS = {
        "unet": smp.Unet,
        "unet++": smp.UnetPlusPlus,
        "deeplabv3+": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
        "pspnet": smp.PSPNet,
        "pan": smp.PAN,
        "transunet": TransUNet,  # Add the TransUNet models here
    }

    # Create segmentation models with pretrained encoder
    if args["models"] == "transunet":
        model = MODELS[args["models"]](len(config.CLASSES))
    else:
        model = MODELS[args["models"]](
            encoder_name=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            classes=len(config.CLASSES),
            activation=config.ACTIVATION,
        )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    # Get train and val dataset instances
    train_dataset = CracksDataset(
        config.X_TRAIN_DIR, config.Y_TRAIN_DIR,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn, model_name=args["models"]),
        class_rgb_values=config.CLASS_RGB_VALUES,
    )

    valid_dataset = CracksDataset(
        config.X_VALID_DIR, config.Y_VALID_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn, model_name=args["models"]),
        class_rgb_values=config.CLASS_RGB_VALUES,
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Define loss function
    loss = config.LOSS

    # Define metrics
    metrics = config.METRIC

    # Define optimizer
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.INIT_LR)])

    # Define learning rate scheduler (not used in this script)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    train_epoch = su.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = su.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for e in tqdm(range(0, config.EPOCHS)):
        # Perform training & validation
        print('\nEpoch: {}'.format(e))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        file_name = f'model_epoch_{e}.pth'
        torch.save(model, os.path.join(args["output"], file_name))

        # Save models if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, os.path.join(args["output"], 'best_model.pth'))
            print('Model saved!')

    print("Evaluation on Test Data:")
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T.to_csv(os.path.join(args["output"], 'train_logs.csv'))

    # Paths to save plots
    iou_plot_path = os.path.join(args["output"], 'iou_score_plot.png')
    dice_plot_path = os.path.join(args["output"], 'dice_loss_plot.png')

    # Plot IoU Score
    plot_metric(train_logs_df, valid_logs_df, 'iou_score', 'IoU Score', 'IoU Score Plot', iou_plot_path)

    # Plot Dice Loss
    plot_metric(train_logs_df, valid_logs_df, 'dice_loss', 'Dice Loss', 'Dice Loss Plot', dice_plot_path)


if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", type=str, default="unet",
                    choices=["unet", "unet++", "deeplabv3+", "fpn", "manet", "linknet", "pspnet", "pan", "transunet"],
                    help="name of segmentation models to train")
    ap.add_argument("-o", "--output", required=True, help="path to the output directory")
    args = vars(ap.parse_args())

    main(args)
