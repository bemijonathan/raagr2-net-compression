import json
import torch
from src.utils.stats import ModelPerformanceMetrics
from src.architecture.model import load_trained_model, class_dice
from src.utils.custom_metric import dice_coef, mean_iou
from src.utils.load_data import BrainDataset
from torch.utils.data import DataLoader


def main():
    # set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    # models to analyze
    models = {
        "base_model": "model/base_model/dlu_net_model_epoch_35.pth",
        "snip_model_10": "model/snip/dlu_net_model_epoch_10.pth",
        "snip_model_20": "model/snip/dlu_net_model_epoch_20.pth",
        "snip_model_30": "model/snip/dlu_net_model_epoch_30.pth",
        "magnitude_10": "model/magnitude/dlu_net_model_epoch_10.pth",
        "magnitude_20": "model/magnitude/dlu_net_model_epoch_20.pth",
        "magnitude_30": "model/magnitude/dlu_net_model_epoch_30.pth",
        "slimmed_10": "model/slimmed/dlu_net_model_epoch_30.pth",
        "slimmed_20": "model/slimmed/dlu_net_model_epoch_30.pth",
        "slimmed_30": "model/slimmed/dlu_net_model_epoch_30.pth",
        "shared": "model/shared/dlu_net_model_epoch_35.pth",
        "reduced_shared":"model/reduced_shared/dlu_net_model_epoch_35.35pth"
    }

    # prepare evaluation data
    dataset = BrainDataset("./data", "val")
    loader = DataLoader(dataset, batch_size=16)
    X_val, y_val = next(iter(loader))
    X_val, y_val = X_val.to(device), y_val.to(device)
    input_shape = X_val.shape

    all_results = {}
    # define metric functions for segmentation
    metric_functions = {
        "mean_iou": mean_iou,
        "dice_coef": dice_coef,
        "c_2": class_dice,
        "c_3": class_dice,
        "c_4": class_dice,
    }

    for name, path in models.items():
        # load and prepare model
        model = load_trained_model(path)
        model.to(device)
        # extract metrics
        mpm = ModelPerformanceMetrics(name)
        _ = mpm.extract_metrics_from_model(
            model,
            input_data=X_val,
            input_shape=input_shape,
            model_path=path,
            eval_data=(X_val, y_val),
            metric_functions=metric_functions
        )
        all_results[name] = mpm.to_dict()

    # save all metrics
    with open("all_model_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("All metrics saved to all_model_metrics.json")


if __name__ == "__main__":
    main()
