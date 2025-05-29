import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create charts directory if it doesn't exist
CHARTS_DIR = 'charts'
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)
    print(f"Created directory: {CHARTS_DIR}")

# Load the data
with open('model_results.json', 'r') as f:
    data = json.load(f)


def save_chart(filename_base):
    """Helper function to save charts in both PNG and PDF formats"""
    png_path = os.path.join(CHARTS_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(CHARTS_DIR, f"{filename_base}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {png_path}")
    print(f"  - Saved: {pdf_path}")


def create_performance_metrics_chart():
    """Create a comprehensive performance metrics comparison chart"""

    # Prepare data
    models = []
    dice_scores = []
    mean_ious = []
    c2_dice = []
    c3_dice = []
    c4_dice = []
    colors = []

    # Define color scheme for different model types
    color_map = {
        'base': '#2E86AB',
        'architectural': '#A23B72',
        'magnitude': '#F18F01',
        'snip': '#C73E1D',
        'depgraph': '#7209B7',
        'finetuned': '#06A77D'
    }

    # Add base model
    models.append('Base RAAGR2-Net')
    dice_scores.append(data['base_model']['metrics']['dice_coef'])
    mean_ious.append(data['base_model']['metrics']['mean_iou'])
    c2_dice.append(data['base_model']['metrics']['class_dice']['c2'])
    c3_dice.append(data['base_model']['metrics']['class_dice']['c3'])
    c4_dice.append(data['base_model']['metrics']['class_dice']['c4'])
    colors.append(color_map['base'])

    # Add architectural modifications - Updated to use correct keys
    arch_models = {
        'Depthwise Shared': 'depthwise_shared',
        'Paired Shared': 'paired_shared_depthwise',
        'Encoder-Decoder': 'enc_con_stats'
    }

    for name, key in arch_models.items():
        if key in data:
            models.append(name)
            dice_scores.append(data[key]['metrics']['dice_coef'])
            mean_ious.append(data[key]['metrics']['mean_iou'])
            c2_dice.append(data[key]['metrics']['class_dice']['c2'])
            c3_dice.append(data[key]['metrics']['class_dice']['c3'])
            c4_dice.append(data[key]['metrics']['class_dice']['c4'])
            colors.append(color_map['architectural'])

    # Add DepGraph pruning results (best performing ratio) - Fixed key name
    depgraph_models = {
        'DepGraph 10%': ('depgraph', '10_percent'),
        'DepGraph 20%': ('depgraph', '20_percent'),
    }

    for name, (key, ratio) in depgraph_models.items():
        if key in data and ratio in data[key]:
            models.append(name)
            dice_scores.append(data[key][ratio]['metrics']['dice_coef'])
            mean_ious.append(data[key][ratio]['metrics']['mean_iou'])
            c2_dice.append(data[key][ratio]['metrics']['class_dice']['c2'])
            c3_dice.append(data[key][ratio]['metrics']['class_dice']['c3'])
            c4_dice.append(data[key][ratio]['metrics']['class_dice']['c4'])
            colors.append(color_map['depgraph'])

    # Add best pruning results (20% with finetuning)
    pruning_models = {
        'Magnitude 20% (FT)': 'magnitude_pruning_finetuned',
        'SNIP 20% (FT)': 'snip_one_shot_finetuned'
    }

    for name, key in pruning_models.items():
        if key in data and '20_percent' in data[key]:
            models.append(name)
            dice_scores.append(data[key]['20_percent']['metrics']['dice_coef'])
            mean_ious.append(data[key]['20_percent']['metrics']['mean_iou'])
            c2_dice.append(data[key]['20_percent']
                           ['metrics']['class_dice']['c2'])
            c3_dice.append(data[key]['20_percent']
                           ['metrics']['class_dice']['c3'])
            c4_dice.append(data[key]['20_percent']
                           ['metrics']['class_dice']['c4'])
            colors.append(color_map['finetuned'])

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics Comparison Across Model Variants',
                 fontsize=16, fontweight='bold')

    # Plot 1: Overall Performance (Dice & IoU)
    x = np.arange(len(models))
    width = 0.35

    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, dice_scores, width,
                    label='Dice Coefficient', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, mean_ious, width,
                    label='Mean IoU', color=colors, alpha=0.6)

    ax1.set_xlabel('Model Variants')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance: Dice Coefficient & Mean IoU')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                 f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                 f'{height2:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Class-wise Performance
    ax2 = axes[0, 1]
    width = 0.25
    x2 = np.arange(len(models))

    bars_c2 = ax2.bar(x2 - width, c2_dice, width,
                      label='NCR/NET (C2)', alpha=0.8)
    bars_c3 = ax2.bar(x2, c3_dice, width, label='ET (C3)', alpha=0.8)
    bars_c4 = ax2.bar(x2 + width, c4_dice, width, label='ED (C4)', alpha=0.8)

    ax2.set_xlabel('Model Variants')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Class-wise Performance Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Performance vs Base Model (Percentage)
    ax3 = axes[1, 0]
    base_dice = data['base_model']['metrics']['dice_coef']
    relative_performance = [(score/base_dice - 1) *
                            100 for score in dice_scores]

    bars3 = ax3.bar(range(len(models)), relative_performance,
                    color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Model Variants')
    ax3.set_ylabel('Performance Change (%)')
    ax3.set_title('Performance Relative to Base Model')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                 f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # Plot 4: Model Type Legend and Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create legend - Updated to include DepGraph
    legend_elements = [
        mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=color_map['base'], label='Base Model'),
        mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=color_map['architectural'], label='Architectural Modifications'),
        mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=color_map['depgraph'], label='DepGraph Pruning'),
        mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=color_map['finetuned'], label='Finetuned Pruning'),
    ]

    ax4.legend(handles=legend_elements, loc='center',
               fontsize=12, title='Model Categories')

    # Add summary statistics
    summary_text = f"""
    Summary Statistics:
    
    Best Overall: {models[np.argmax(dice_scores)]}
    Dice: {max(dice_scores):.4f}
    
    Best Efficiency: Depthwise Shared
    43.4% parameter reduction
    0.12% performance loss
    
    Key Finding: Architectural modifications
    outperform traditional pruning
    """

    ax4.text(0.1, 0.3, summary_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    save_chart('performance_metrics_chart')
    plt.show()


def create_computational_benchmarks_chart():
    """Create computational benchmarks comparison chart"""

    # Prepare data
    models = []
    params = []
    sizes = []
    inference_times = []
    flops = []
    gpu_memory = []
    colors = []

    color_map = {
        'base': '#2E86AB',
        'architectural': '#A23B72',
        'depgraph': '#7209B7',
        'finetuned': '#06A77D'
    }

    # Base model
    models.append('Base RAAGR2-Net')
    params.append(data['base_model']['model_stats']
                  ['num_params'] / 1e6)  # Convert to millions
    sizes.append(data['base_model']['model_stats']['model_size_mb'])
    inference_times.append(
        data['base_model']['model_stats']['avg_inference_time_ms'])
    flops.append(data['base_model']['model_stats']
                 ['flops'] / 1e9)  # Convert to GFLOPs
    gpu_memory.append(data['base_model']['memory_stats']['peak_gpu_memory_gb'])
    colors.append(color_map['base'])

    # Architectural modifications - Updated to use correct keys
    arch_models = {
        'Depthwise Shared': ('depthwise_shared', color_map['architectural']),
        'Paired Shared': ('paired_shared_depthwise', color_map['architectural']),
        'Encoder-Decoder': ('enc_con_stats', color_map['architectural'])
    }

    for name, (key, color) in arch_models.items():
        if key in data:
            models.append(name)
            params.append(data[key]['model_stats']['num_params'] / 1e6)
            sizes.append(data[key]['model_stats']['model_size_mb'])
            inference_times.append(
                data[key]['model_stats']['avg_inference_time_ms'])
            flops.append(data[key]['model_stats']['flops'] / 1e9)
            gpu_memory.append(data[key]['memory_stats']['peak_gpu_memory_gb'])
            colors.append(color)

    # DepGraph pruning
    depgraph_models = {
        'DepGraph 10%': ('depgraph', '10_percent'),
        'DepGraph 20%': ('depgraph', '20_percent'),
    }

    for name, (key, ratio) in depgraph_models.items():
        if key in data and ratio in data[key]:
            models.append(name)
            params.append(data[key][ratio]['model_stats']['num_params'] / 1e6)
            sizes.append(data[key][ratio]['model_stats']['model_size_mb'])
            inference_times.append(
                data[key][ratio]['model_stats']['avg_inference_time_ms'])
            flops.append(data[key][ratio]['model_stats']['flops'] / 1e9)
            gpu_memory.append(
                data[key][ratio]['memory_stats']['peak_gpu_memory_gb'])
            colors.append(color_map['depgraph'])

    # Finetuned pruning
    pruning_models = {
        'Magnitude 20% (FT)': 'magnitude_pruning_finetuned',
        'SNIP 20% (FT)': 'snip_one_shot_finetuned'
    }

    for name, key in pruning_models.items():
        if key in data and '20_percent' in data[key]:
            models.append(name)
            params.append(data[key]['20_percent']
                          ['model_stats']['num_params'] / 1e6)
            sizes.append(data[key]['20_percent']
                         ['model_stats']['model_size_mb'])
            inference_times.append(
                data[key]['20_percent']['model_stats']['avg_inference_time_ms'])
            flops.append(data[key]['20_percent']['model_stats']['flops'] / 1e9)
            gpu_memory.append(data[key]['20_percent']
                              ['memory_stats']['peak_gpu_memory_gb'])
            colors.append(color_map['finetuned'])

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Computational Benchmarks Comparison',
                 fontsize=16, fontweight='bold')

    # Plot 1: Parameters
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(models)), params, color=colors, alpha=0.8)
    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_title('Model Parameters')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}M', ha='center', va='bottom', fontsize=9)

    # Plot 2: Model Size
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(models)), sizes, color=colors, alpha=0.8)
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}MB', ha='center', va='bottom', fontsize=9)

    # Plot 3: Inference Time
    ax3 = axes[0, 2]
    bars3 = ax3.bar(range(len(models)), inference_times,
                    color=colors, alpha=0.8)
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Average Inference Time')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)

    # Plot 4: FLOPs
    ax4 = axes[1, 0]
    bars4 = ax4.bar(range(len(models)), flops, color=colors, alpha=0.8)
    ax4.set_ylabel('FLOPs (GFLOPs)')
    ax4.set_title('Computational Complexity')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{height:.1f}G', ha='center', va='bottom', fontsize=9)

    # Plot 5: GPU Memory
    ax5 = axes[1, 1]
    bars5 = ax5.bar(range(len(models)), gpu_memory, color=colors, alpha=0.8)
    ax5.set_ylabel('GPU Memory (GB)')
    ax5.set_title('Peak GPU Memory Usage')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels(models, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}GB', ha='center', va='bottom', fontsize=9)

    # Plot 6: Efficiency Summary
    ax6 = axes[1, 2]

    # Calculate efficiency metrics (relative to base)
    base_params = params[0]
    base_time = inference_times[0]

    param_reduction = [(base_params - p) / base_params * 100 for p in params]
    time_improvement = [(base_time - t) / base_time *
                        100 for t in inference_times]

    x_pos = np.arange(len(models))
    width = 0.35

    bars_param = ax6.bar(x_pos - width/2, param_reduction, width,
                         label='Parameter Reduction (%)', alpha=0.8)
    bars_time = ax6.bar(x_pos + width/2, time_improvement, width,
                        label='Speed Improvement (%)', alpha=0.8)

    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('Efficiency Improvements vs Base')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(models, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)

    plt.tight_layout()
    save_chart('computational_benchmarks_chart')
    plt.show()


def create_reduction_analysis_heatmap():
    """Create a comprehensive reduction analysis heatmap"""

    # Prepare data for heatmap
    models = ['Depthwise Shared', 'Paired Shared', 'Encoder-Decoder Shared']
    metrics = ['Parameter\nReduction', 'Size\nReduction', 'FLOPs\nReduction',
               'GPU Memory\nReduction', 'Time\nReduction']

    # Calculate reduction percentages
    base_stats = data['base_model']
    base_params = base_stats['model_stats']['num_params']
    base_size = base_stats['model_stats']['model_size_mb']
    base_flops = base_stats['model_stats']['flops']
    base_gpu = base_stats['memory_stats']['peak_gpu_memory_gb']
    base_time = base_stats['model_stats']['avg_inference_time_ms']

    reduction_data = []

    # Calculate for each architectural modification - Updated to use correct keys
    arch_keys = ['depthwise_shared',
                 'paired_shared_depthwise', 'enc_con_stats']

    for key in arch_keys:
        if key in data:
            stats = data[key]
            param_red = (
                base_params - stats['model_stats']['num_params']) / base_params * 100
            size_red = (base_size - stats['model_stats']
                        ['model_size_mb']) / base_size * 100
            flops_red = (
                base_flops - stats['model_stats']['flops']) / base_flops * 100
            gpu_red = (base_gpu - stats['memory_stats']
                       ['peak_gpu_memory_gb']) / base_gpu * 100
            time_red = (base_time - stats['model_stats']
                        ['avg_inference_time_ms']) / base_time * 100

            reduction_data.append(
                [param_red, size_red, flops_red, gpu_red, time_red])

    reduction_matrix = np.array(reduction_data)

    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap
    im = ax1.imshow(reduction_matrix, cmap='RdYlGn',
                    aspect='auto', vmin=-30, vmax=50)

    # Set ticks and labels
    ax1.set_xticks(np.arange(len(metrics)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(metrics)
    ax1.set_yticklabels(models)

    # Rotate the tick labels and set their alignment
    plt.setp(ax1.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax1.text(j, i, f'{reduction_matrix[i, j]:.1f}%',
                            ha="center", va="center", color="black", fontweight='bold')

    ax1.set_title(
        "Reduction Analysis Heatmap\n(% Improvement vs Base Model)", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Reduction Percentage (%)', rotation=270, labelpad=20)

    # Performance vs Efficiency Scatter Plot
    ax2.set_xlim(0, 50)
    ax2.set_ylim(-1, 1)

    # Calculate performance change for each model
    base_dice = data['base_model']['metrics']['dice_coef']

    for i, key in enumerate(arch_keys):
        if key in data:
            dice_change = (data[key]['metrics']
                           ['dice_coef'] - base_dice) / base_dice * 100
            param_reduction = reduction_matrix[i, 0]  # Parameter reduction

            # Size of bubble represents overall efficiency
            # Average of size and FLOPs reduction
            size = (reduction_matrix[i, 1] + reduction_matrix[i, 2]) / 2
            size = max(100, size * 10)  # Scale for visibility

            ax2.scatter(param_reduction, dice_change, s=size, alpha=0.7,
                        label=models[i], c=f'C{i}')

            # Add model name annotation
            ax2.annotate(models[i], (param_reduction, dice_change),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Parameter Reduction (%)')
    ax2.set_ylabel('Performance Change (%)')
    ax2.set_title(
        'Efficiency vs Performance Trade-off\n(Bubble size = Combined Reduction)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add quadrant labels
    ax2.text(40, 0.8, 'High Efficiency\nHigh Performance', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax2.text(40, -0.8, 'High Efficiency\nLow Performance', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout()
    save_chart('reduction_analysis_heatmap')
    plt.show()


def create_comprehensive_comparison():
    """Create a comprehensive comparison chart combining all metrics"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Model Comparison Dashboard',
                 fontsize=18, fontweight='bold')

    # Data preparation - Updated to include DepGraph and use correct keys
    models = ['Base', 'Depthwise\nShared', 'Paired\nShared', 'Encoder-Dec\nShared',
              'DepGraph\n10%', 'Magnitude\n20% (FT)', 'SNIP\n20% (FT)']

    # Collect data - Updated to use correct keys
    model_keys = ['base_model', 'depthwise_shared', 'paired_shared_depthwise',
                  'enc_con_stats', ('depgraph', '10_percent'),
                  'magnitude_pruning_finetuned', 'snip_one_shot_finetuned']

    dice_scores = []
    param_counts = []
    model_sizes = []
    gpu_memory = []

    for i, key in enumerate(model_keys):
        if isinstance(key, tuple):
            # Handle DepGraph case
            main_key, ratio = key
            if main_key in data and ratio in data[main_key]:
                dice_scores.append(data[main_key][ratio]
                                   ['metrics']['dice_coef'])
                param_counts.append(
                    data[main_key][ratio]['model_stats']['num_params'] / 1e6)
                model_sizes.append(data[main_key][ratio]
                                   ['model_stats']['model_size_mb'])
                gpu_memory.append(data[main_key][ratio]
                                  ['memory_stats']['peak_gpu_memory_gb'])
        elif key in data:
            if 'finetuned' in key:
                # Finetuned models have nested structure
                dice_scores.append(
                    data[key]['20_percent']['metrics']['dice_coef'])
                param_counts.append(
                    data[key]['20_percent']['model_stats']['num_params'] / 1e6)
                model_sizes.append(
                    data[key]['20_percent']['model_stats']['model_size_mb'])
                gpu_memory.append(data[key]['20_percent']
                                  ['memory_stats']['peak_gpu_memory_gb'])
            else:
                dice_scores.append(data[key]['metrics']['dice_coef'])
                param_counts.append(
                    data[key]['model_stats']['num_params'] / 1e6)
                model_sizes.append(data[key]['model_stats']['model_size_mb'])
                gpu_memory.append(
                    data[key]['memory_stats']['peak_gpu_memory_gb'])

    # Define colors for model types - Updated to include DepGraph
    colors = ['#2E86AB', '#A23B72', '#A23B72',
              '#A23B72', '#7209B7', '#06A77D', '#06A77D']

    # Plot 1: Performance vs Model Size
    ax1.scatter(model_sizes, dice_scores, s=200, c=colors, alpha=0.8)
    for i, model in enumerate(models):
        ax1.annotate(model, (model_sizes[i], dice_scores[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.set_xlabel('Model Size (MB)')
    ax1.set_ylabel('Dice Coefficient')
    ax1.set_title('Performance vs Model Size')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter Reduction Comparison
    base_params = param_counts[0]
    param_reductions = [(base_params - p) / base_params *
                        100 for p in param_counts]

    bars = ax2.bar(range(len(models)), param_reductions,
                   color=colors, alpha=0.8)
    ax2.set_ylabel('Parameter Reduction (%)')
    ax2.set_title('Parameter Reduction vs Base Model')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Memory Efficiency
    ax3.bar(range(len(models)), gpu_memory, color=colors, alpha=0.8)
    ax3.set_ylabel('Peak GPU Memory (GB)')
    ax3.set_title('GPU Memory Usage')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overall Efficiency Score
    # Create a composite efficiency score
    efficiency_scores = []
    for i in range(len(models)):
        # Normalize metrics (higher is better for all)
        dice_norm = dice_scores[i] / max(dice_scores)
        param_red_norm = max(
            0, param_reductions[i]) / max(param_reductions) if max(param_reductions) > 0 else 0
        memory_eff = (max(gpu_memory) - gpu_memory[i]) / max(gpu_memory)

        # Weighted combination
        efficiency_score = 0.4 * dice_norm + 0.4 * param_red_norm + 0.2 * memory_eff
        efficiency_scores.append(efficiency_score * 100)

    bars4 = ax4.bar(range(len(models)), efficiency_scores,
                    color=colors, alpha=0.8)
    ax4.set_ylabel('Overall Efficiency Score')
    ax4.set_title('Composite Efficiency Ranking')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    # Add legend - Updated to include DepGraph
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='Base Model'),
        mpatches.Patch(color='#A23B72', label='Architectural Modifications'),
        mpatches.Patch(color='#7209B7', label='DepGraph Pruning'),
        mpatches.Patch(color='#06A77D', label='Finetuned Pruning')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    save_chart('comprehensive_comparison_dashboard')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GENERATING COMPREHENSIVE CHARTS FOR MODEL ANALYSIS")
    print("=" * 60)

    print("\n📊 Creating Performance Metrics Chart...")
    create_performance_metrics_chart()

    print("\n⚡ Creating Computational Benchmarks Chart...")
    create_computational_benchmarks_chart()

    print("\n🔥 Creating Reduction Analysis Heatmap...")
    create_reduction_analysis_heatmap()

    print("\n📈 Creating Comprehensive Comparison Dashboard...")
    create_comprehensive_comparison()

    print("\n" + "=" * 60)
    print("✅ ALL CHARTS CREATED SUCCESSFULLY!")
    print("=" * 60)

    print(f"\n📁 All files organized in: ./{CHARTS_DIR}/")
    print("\n📋 Generated Files:")

    chart_files = [
        "performance_metrics_chart",
        "computational_benchmarks_chart",
        "reduction_analysis_heatmap",
        "comprehensive_comparison_dashboard"
    ]

    for i, filename in enumerate(chart_files, 1):
        print(f"   {i}. {filename}.png")
        print(f"   {i}. {filename}.pdf")

    print(f"\n🎯 Total: {len(chart_files) * 2} files (PNG + PDF versions)")
    print(f"📍 Location: {os.path.abspath(CHARTS_DIR)}")

    print("\n" + "=" * 60)
    print("🔧 USAGE IN LATEX:")
    print("=" * 60)
    print("\\begin{figure}[H]")
    print("\\centering")
    print(
        "\\includegraphics[width=\\textwidth]{charts/performance_metrics_chart.pdf}")
    print("\\caption{Performance metrics comparison}")
    print("\\label{fig:performance_metrics}")
    print("\\end{figure}")
    print("=" * 60)
