import torch
from src.utils.training import resume_training
from src.architecture.enc_dec_shared_model import get_initial_model, load_trained_model, class_dice, dice_coef, mean_iou
from src.architecture.model import load_trained_model as ltm
from src.utils.custom_loss import Weighted_BCEnDice_loss
from src.utils.load_data import BrainDataset
from torch.utils.data import DataLoader
from src.CONFIGS import batch_size
from src.utils.stats import ModelPerformanceMetrics, ModelComparison
import numpy as np

device = torch.device("mps")


# Comment out the real data loading
# data_path = "data/"
# val_dataset = BrainDataset(data_path, "val")
# train_dataset = BrainDataset(data_path, "train")
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Dummy data for testing


def test_dummy_training():
    print("\n===== DUMMY TRAINING TEST =====")
    model = get_initial_model()
    model.train()
    dummy_input = torch.randn(batch_size, 4, 192, 192).to(device)
    dummy_target = torch.randint(
        0, 2, (batch_size, 5, 192, 192)).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_function = Weighted_BCEnDice_loss
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = loss_function(output, dummy_target)
    print(f"Dummy loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    print("Dummy training step completed successfully.")


def main():
    model = get_initial_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # train for one epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.1)

    new_model = resume_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=Weighted_BCEnDice_loss,
        device=device,
        resume_checkpoint_path="",
        starting_epoch=0,
        num_epochs=10,
        checkpoint_interval=5,
        model_save_dir='model/enc_dec_shared_model',
        experiment_name="enc_dec_shared_model",
        validate_every=10,
    )

    torch.save(
        new_model, 'model/enc_dec_shared_model/enc_dec_shared_model_epoch_10.pth')


def test_model():

    # 1. First create a fresh model instance
    # model = torch.load(
    #     "mlruns/820203924686178493/40d84b246dd1475a93e8fb5c244ae452/artifacts/checkpoints/dlu_net_model_epoch_15.pth",
    #     map_location=device,
    #     weights_only=False
    # )
    # model = model[0]
    model = load_trained_model(
        "mlruns/820203924686178493/3f3f50dcc84a40efa3665e02a104d67f/artifacts/checkpoints/dlu_net_model_epoch_40.pth")
    # 5. Set model to evaluation mode
    model.eval()
    print("Model loaded successfully and set to evaluation mode")

    val_images, val_masks = next(iter(val_loader))

    with torch.no_grad():
        results = []

        with torch.no_grad():

            for i in range(len(val_images)):
                # Get the first sample
                sample_image = val_images[i:i + 1].to(device)
                sample_mask = val_masks[i:i + 1].to(device)

                # Get prediction
                prediction = model(sample_image)
                thresholded_pred = (prediction > 0.2).float()

                # Use the prediction directly for dice calculation instead of calling evaluate_dice_scores
                tc_dice = class_dice(prediction, sample_mask, 2).item()
                ec_dice = class_dice(prediction, sample_mask, 3).item()
                wt_dice = class_dice(prediction, sample_mask, 4).item()

                dice_score_main = dice_coef(prediction, sample_mask)
                mean_iou_score = mean_iou(prediction, sample_mask)

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

        # # get the IQR of the results

        tc_dice_values = [x[0] for x in results]
        ec_dice_values = [x[1] for x in results]
        wt_dice_values = [x[2] for x in results]
        dice_score_main_values = [x[3] for x in results]
        mean_iou_values = [x[4] for x in results]

        print(
            f"IQR of Tumor Core Dice: {np.percentile(tc_dice_values, 75) - np.percentile(tc_dice_values, 25)}")
        print(
            f"IQR of Enhancing Tumor Dice: {np.percentile(ec_dice_values, 75) - np.percentile(ec_dice_values, 25)}")
        print(
            f"IQR of Whole Tumor Dice: {np.percentile(wt_dice_values, 75) - np.percentile(wt_dice_values, 25)}")


def statistics():
    shared_model = load_trained_model(
        "mlruns/820203924686178493/d066cef71fe742a48417b685a8b6a8d6/artifacts/checkpoints/dlu_net_model_epoch_60.pth")
    original_model = ltm("model/dlu_net_model_epoch_58.pth")

    shared_model = shared_model.to(device)
    original_model = original_model.to(device)

    # Verify shared weights
    verify_weight_sharing(shared_model)

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


def verify_weight_sharing(model):
    """Verify that weights are correctly shared between encoder and decoder paths"""
    print("\n===== WEIGHT SHARING VERIFICATION =====")

    # Check if encoder and decoder ReASPP3 blocks share the same weights
    print("\nChecking ReASPP3 shared weights:")
    # Compare weights between encoder block 1 and decoder block 2
    enc1_weight_id = id(model.shared_block1.shared_dw_weight)
    dec2_weight_id = id(model.dec2.shared_dw_weight)
    print(
        f"Encoder block 1 and decoder block 2 weights shared: {enc1_weight_id == dec2_weight_id}")

    # Compare weights between encoder block 2 and decoder block 3
    enc2_weight_id = id(model.shared_block2.shared_dw_weight)
    dec3_weight_id = id(model.dec3.shared_dw_weight)
    print(
        f"Encoder block 2 and decoder block 3 weights shared: {enc2_weight_id == dec3_weight_id}")

    # Compare weights between encoder block 3 and decoder block 4
    enc3_weight_id = id(model.shared_block3.shared_dw_weight)
    dec4_weight_id = id(model.dec4.shared_dw_weight)
    print(
        f"Encoder block 3 and decoder block 4 weights shared: {enc3_weight_id == dec4_weight_id}")

    # Compare weights between encoder block 4 and decoder block 5
    enc4_weight_id = id(model.shared_block4.shared_dw_weight)
    dec5_weight_id = id(model.dec5.shared_dw_weight)
    print(
        f"Encoder block 4 and decoder block 5 weights shared: {enc4_weight_id == dec5_weight_id}")

    # Calculate parameter savings
    total_parameters = sum(p.numel() for p in model.parameters())
    # Before accounting for sharing
    unique_parameters = sum(p.numel() for p in model.parameters())

    # Subtract shared parameters from unique count
    # Each shared weight tensor is counted twice, so subtract once
    for i in range(1, 5):
        block = getattr(model, f"shared_block{i}")
        unique_parameters -= block.shared_dw_weight.numel()
        unique_parameters -= block.shared_dw_bias.numel()

    parameter_savings = 1 - (unique_parameters / total_parameters)
    print(f"\nTotal parameters (with duplicates): {total_parameters}")
    print(f"Unique parameters (after sharing): {unique_parameters}")
    print(f"Parameter sharing efficiency: {parameter_savings * 100:.2f}%")


if __name__ == "__main__":
    test_dummy_training()
    # main()
    # test_model()
    # statistics()
