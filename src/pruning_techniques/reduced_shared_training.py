import torch
from src.utils.training import resume_training
from src.architecture.reduce_shared_model import get_initial_model, load_trained_model, evaluate_dice_scores, class_dice, arrange_img, dice_coef, mean_iou
from src.architecture.model import load_trained_model as ltm
from src.utils.custom_loss import Weighted_BCEnDice_loss
from src.utils.load_data import BrainDataset, get_data_loaders
from torch.utils.data import DataLoader
from src.CONFIGS import batch_size
from src.utils.stats import ModelPerformanceMetrics, ModelComparison
import numpy as np

device = torch.device("cuda")

train_loader, val_loader, test_loader = get_data_loaders("data")


def main():
    model = get_initial_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # train for one epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)


    new_model = resume_training(
        model=model,
        optimizer = optimizer,
        scheduler = scheduler,
        train_loader = train_loader,
        val_loader = val_loader,
        loss_function = Weighted_BCEnDice_loss,
        device = device,
        resume_checkpoint_path ="",
        starting_epoch=0,
        num_epochs=35,
        checkpoint_interval=5,
        model_save_dir='model/reduced_shared',
        experiment_name="reduced_shared_weights_model",
        validate_every=1,
    )

    torch.save(new_model, 'model/reduce_shared/shared_dlu_model_epoch_40.pth')

def evaluate_model(model, test_loader, batch_size=8):
    model.eval()

    results = []

    with torch.no_grad():
        for val_images, val_masks in test_loader:
            # Get the first sample
            sample_image = val_images.to(device)
            sample_mask = val_masks.to(device)

            # Get prediction
            prediction = model(sample_image)
            thresholded_pred = (prediction > 0.2).float()

            # Use the prediction directly for dice calculation
            tc_dice = class_dice(thresholded_pred, sample_mask, 2).item()
            ec_dice = class_dice(thresholded_pred, sample_mask, 3).item()
            wt_dice = class_dice(thresholded_pred, sample_mask, 4).item()

            dice_score_main = dice_coef(thresholded_pred, sample_mask).item()
            mean_iou_score = mean_iou(thresholded_pred, sample_mask).item()

            results.append([tc_dice, ec_dice, wt_dice,
                            dice_score_main, mean_iou_score])
        print(
            f"Sample Dice Scores - Tumor Core: {tc_dice:.4f}, "
            f"Enhancing Tumor: {ec_dice:.4f}, "
            f"Whole Tumor: {wt_dice:.4f}"
        )

    # Get average of the results
    print(results)
    avg_tc_dice = sum([x[0] for x in results]) / len(results)
    avg_ec_dice = sum([x[1] for x in results]) / len(results)
    avg_wt_dice = sum([x[2] for x in results]) / len(results)
    avg_dice_score_main = sum([x[3] for x in results]) / len(results)
    avg_mean_iou = sum([x[4] for x in results]) / len(results)

    objec =  {
        "mean_iou": avg_mean_iou,
        "dice_coef": avg_dice_score_main,
        "c_2": avg_tc_dice.real,
        "c_3": avg_ec_dice.real,
        "c_4": avg_wt_dice.real,
    }
    print(objec)
    return objec

def test_model():

    model = load_trained_model("mlruns/452079217653245001/f04a254ed9eb4688936017b4d1deb971/artifacts/checkpoints/dlu_net_model_epoch_35.pth")
    # 5. Set model to evaluation mode
    evaluate_model(model=model, test_loader=test_loader)


def statistics():
    shared_model = load_trained_model("mlruns/452079217653245001/f04a254ed9eb4688936017b4d1deb971/artifacts/checkpoints/dlu_net_model_epoch_35.pth")
    original_model = ltm("model/base_model/dlu_net_model_epoch_35.pth")

    shared_model = shared_model.to(device)
    original_model = original_model.to(device)


    X_val, y_val = next(iter(val_loader))
    X_val, y_val = X_val.to(device), y_val.to(device)


    # 5. compare the current base model vs this model
    print("\n===== PERFORMANCE METRICS =====")

    # Create and save comparison metrics
    original_model_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    original_metrics = original_model_metrics.extract_metrics_from_model(
        original_model
    )

    pruned_model_metrics = ModelPerformanceMetrics("Shared_DLU_Net")
    pruned_metrics = pruned_model_metrics.extract_metrics_from_model(
        shared_model
    )

    pruned_model_metrics.benchmark_inference_speed(shared_model, X_val)
    original_model_metrics.benchmark_inference_speed(original_model, X_val)


    model_comparison = ModelComparison(
        original_model_metrics, pruned_model_metrics)

    model_comparison.calculate_speedup()
    model_comparison.print_summary()


if __name__ == "__main__":
    main()
    # test_model()
    # statistics()
