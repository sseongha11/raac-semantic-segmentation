# USAGE
# python inference.py --output /content/drive/MyDrive/raac/output/unet/
# python inference.py --output /content/drive/MyDrive/raac/output/unetpp/
# python inference.py --output /content/drive/MyDrive/raac/output/deeplabv3p/
# python inference.py --output /content/drive/MyDrive/raac/output/fpn/
# python inference.py --output /content/drive/MyDrive/raac/output/linknet/
# python inference.py --output /content/drive/MyDrive/raac/output/pspnet/
# python inference.py --output /content/drive/MyDrive/raac/output/pan/

import os
import random
import cv2
import numpy as np
import argparse
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as su

from torch.utils.data import DataLoader
from raac import config
from raac.dataset import CracksDataset
from raac.helper_functions import (
    colour_code_segmentation, crop_image, get_preprocessing,
    get_validation_augmentation, reverse_one_hot, visualize
)


def load_model(output_dir):
    model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=config.DEVICE)
        print('Loaded the models from this run.')
        return model
    else:
        print('No saved models checkpoint found. Exiting...')
        exit()


def evaluate_model(model, dataloader):
    loss = config.LOSS
    metrics = config.METRIC
    test_epoch = su.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )
    return test_epoch.run(dataloader)


def visualize_predictions(model, dataset, dataset_vis, output_dir):
    model.eval()  # Set the model to evaluation mode
    for idx in range(len(dataset)):
        image, gt_mask = dataset[idx]
        image_vis = crop_image(dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
        with torch.no_grad():  # Disable gradient calculations
            pred_mask = model(x_tensor)  # Use the forward method instead of predict
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_building_heatmap = pred_mask[:, :, config.CLASS_NAME.index('crack')]
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), config.CLASS_RGB_VALUES)
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), config.CLASS_RGB_VALUES)

        # Resize masks to match image dimensions
        pred_mask = cv2.resize(pred_mask, (image_vis.shape[1], image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (image_vis.shape[1], image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)

        output_path = os.path.join(output_dir, f"sample_pred_{idx}.png")
        concatenated_image = np.hstack([image_vis, gt_mask, pred_mask])
        cv2.imwrite(output_path, concatenated_image[:, :, ::-1])

        visualize(
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            predicted_building_heatmap=pred_building_heatmap
        )


def main(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_model = load_model(output_dir)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    # Get test dataset instance
    test_dataset = CracksDataset(
        config.X_TEST_DIR, config.Y_TEST_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn, model_name="transunet"),
        class_rgb_values=config.CLASS_RGB_VALUES,
    )

    test_dataloader = DataLoader(test_dataset)
    valid_logs = evaluate_model(best_model, test_dataloader)

    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
    print(f"F1 Score: {valid_logs['fscore']:.4f}")
    print(f"Precision: {valid_logs['precision']:.4f}")
    print(f"Recall: {valid_logs['recall']:.4f}")

    test_dataset_vis = CracksDataset(
        config.X_TEST_DIR, config.Y_TEST_DIR,
        augmentation=get_validation_augmentation(),
        class_rgb_values=config.CLASS_RGB_VALUES,
    )

    random_idx = random.randint(0, len(test_dataset_vis) - 1)
    image, mask = test_dataset_vis[random_idx]

    visualize(
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), config.CLASS_RGB_VALUES),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    visualize_predictions(best_model, test_dataset, test_dataset_vis, output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="path to the output directory")
    args = vars(ap.parse_args())
    main(args["output"])
