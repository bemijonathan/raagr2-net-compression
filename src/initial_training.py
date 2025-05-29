import torch
from architecture.model import DLUNet,  load_trained_model
from utils.custom_loss import Weighted_BCEnDice_loss
from architecture.model import mean_iou, class_dice
from utils.custom_metric import dice_coef
from utils.load_data import BrainDataset, get_data_loaders
from utils.training import resume_training
from utils.stats import ModelPerformanceMetrics, ModelComparison

# Define the directory containing the data
train_data = "..\data"

batch_size = 8
num_epochs = 10




# Initialize model, optimizer and loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, val_loader, test_loader = get_data_loaders("data")

def train():
    print(f"Using device: {device}")
    model = DLUNet(in_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.1)
    criterion = Weighted_BCEnDice_loss

    model, best_val_loss = resume_training(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=criterion,
        device=device,
        resume_checkpoint_path='model/magnitude/magnitude_pruned_model_20.pth',
        starting_epoch=0,
        num_epochs=20,
        checkpoint_interval=1,
        model_save_dir='model/magnitude/',
        validate_every=1,
        experiment_name='magnitude'
    )


def model_results():
    # get average accuracy of the model for the entire validation set

    import numpy as np
    results = []

    with torch.no_grad():
        model = load_trained_model("model/base_model/dlu_net_model_epoch_35.pth")
        print(model)
        for val_images, val_masks in test_loader:
            # Get the first sample
            sample_image = val_images.to(device)
            sample_mask = val_masks.to(device)

            # Get prediction
            prediction = model(sample_image)
            thresholded_pred = (prediction > 0.2).float()

            # Use the prediction directly for dice calculation instead of calling evaluate_dice_scores
            tc_dice = class_dice(thresholded_pred, sample_mask, 2).item()
            ec_dice = class_dice(thresholded_pred, sample_mask, 3).item()
            wt_dice = class_dice(thresholded_pred, sample_mask, 4).item()

            dice_score_main = dice_coef(thresholded_pred, sample_mask).item()
            mean_iou_score = mean_iou(thresholded_pred, sample_mask).item()

            results.append([tc_dice, ec_dice, wt_dice,
                            dice_score_main, mean_iou_score])
        print(
            f"Sample Dice Scores - Tumor Core: {tc_dice:.4f}, Enhancing Tumor: {ec_dice:.4f}, Whole Tumor: {wt_dice:.4f}")

    # get average of the results
    print(results)
    avg_tc_dice = sum([x[0] for x in results]) / len(results)
    avg_ec_dice = sum([x[1] for x in results]) / len(results)
    avg_wt_dice = sum([x[2] for x in results]) / len(results)
    avg_dice_score_main = sum([x[3] for x in results]) / len(results)
    avg_mean_iou = sum([x[4] for x in results]) / len(results)

    print(
        f"Average Scores - Dice Score Main: {avg_dice_score_main:.4f}, Mean IoU: {avg_mean_iou:.4f}, Tumor Core: {avg_tc_dice:.4f}, Enhancing Tumor: {avg_ec_dice:.4f}, Whole Tumor: {avg_wt_dice:.4f}")



def statistics():
    train_loader, val_loader, test_loader = get_data_loaders("data")


    original_model = load_trained_model("model/base_model/dlu_net_model_epoch_15.pth")
    original_model_2 = load_trained_model("model/base_model/dlu_net_model_epoch_15.pth")

    original_model = original_model.to(device)
    original_model_2 = original_model_2.to(device)


    X_val, y_val = next(iter(test_loader))
    X_val, y_val = X_val.to(device), y_val.to(device)


    # 5. compare the current base model vs this model
    print("\n===== PERFORMANCE METRICS =====")

    # Create and save comparison metrics
    original_model_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    original_metrics = original_model_metrics.extract_metrics_from_model(
        original_model
    )

    original_model_metrics_2 = ModelPerformanceMetrics("Original_DLU_Net")
    original_metrics_2 = original_model_metrics_2.extract_metrics_from_model(
        original_model_2
    )

    original_model_metrics.benchmark_inference_speed(original_model, X_val)
    original_model_metrics_2.benchmark_inference_speed(original_model_2, X_val)


    model_comparison = ModelComparison(
        original_model_metrics, original_model_metrics_2)

    model_comparison.calculate_speedup()
    model_comparison.print_summary()




if __name__ == "__main__":
    train()
    # statistics()
    # model_results()
