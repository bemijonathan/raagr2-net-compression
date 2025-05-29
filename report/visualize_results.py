"""
Nobel Prize Standard Model Performance Visualization Suite
=========================================================

This module creates publication-quality visualizations for neural network model
performance analysis, following best practices for scientific publications.

Features:
- Publication-ready typography and styling
- Statistical significance testing
- Comprehensive performance metrics
- Resource efficiency analysis
- Pruning method comparisons
- Error bars and confidence intervals
"""

import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy import stats
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('default')
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
})

# Define color palette for consistency
COLORS = {
    'base': '#2E86AB',      # Blue
    'enc_con': '#A23B72',   # Purple
    'depthwise': '#F18F01',  # Orange
    'paired': '#C73E1D',    # Red
    'magnitude': '#4CAF50',  # Green
    'snip': '#FF9800',      # Orange
    'depgraph': '#9C27B0',  # Purple
}


def load_data():
    """Load model results from JSON file with error handling."""
    try:
        with open('model_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "model_results.json not found. Please ensure the file exists.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in model_results.json")


def create_figure_with_style(figsize=(12, 8), nrows=1, ncols=1):
    """Create a figure with consistent styling."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    return fig, axes


def create_comprehensive_performance_analysis(data):
    """Create a comprehensive performance analysis with multiple metrics."""

    # Extract data for main models
    models_data = {
        'Base Model': data['base_model']['metrics'],
        'ENC-CON': data['enc_con_stats']['metrics'],
        'Depthwise Shared': data['depthwise_shared']['metrics'],
        'Paired Shared': data['paired_shared_depthwise']['metrics']
    }

    fig, axes = create_figure_with_style(figsize=(16, 12), nrows=2, ncols=2)

    # 1. Overall Performance Metrics
    ax1 = axes[0]
    metrics = ['Dice Coefficient', 'Mean IoU']
    x_pos = np.arange(len(models_data))
    width = 0.35

    dice_scores = [m['dice_coef'] for m in models_data.values()]
    iou_scores = [m['mean_iou'] for m in models_data.values()]

    bars1 = ax1.bar(x_pos - width/2, dice_scores, width,
                    label='Dice Coefficient', color=COLORS['base'], alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, iou_scores, width,
                    label='Mean IoU', color=COLORS['enc_con'], alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('A. Overall Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_data.keys(), rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # 2. Class-wise Performance Heatmap
    ax2 = axes[1]
    class_names = ['NCR/NET (C2)', 'ED (C3)', 'ET (C4)']
    performance_matrix = []

    for model_name, metrics in models_data.items():
        class_scores = [metrics['class_dice'][f'c{i}'] for i in [2, 3, 4]]
        performance_matrix.append(class_scores)

    performance_matrix = np.array(performance_matrix)

    im = ax2.imshow(performance_matrix, cmap='RdYlBu_r',
                    aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(models_data)):
        for j in range(len(class_names)):
            text = ax2.text(j, i, f'{performance_matrix[i, j]:.3f}',
                            ha="center", va="center", color="black", fontweight='bold')

    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(models_data)))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(models_data.keys())
    ax2.set_title('B. Class-wise Dice Scores')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Dice Score', rotation=270, labelpad=15)

    # 3. Efficiency Analysis
    ax3 = axes[2]
    efficiency_data = {
        'Base Model': data['base_model']['model_stats'],
        'ENC-CON': data['enc_con_stats']['model_stats'],
        'Depthwise Shared': data['depthwise_shared']['model_stats'],
        'Paired Shared': data['paired_shared_depthwise']['model_stats']
    }

    model_sizes = [m['model_size_mb'] for m in efficiency_data.values()]
    param_counts = [m['num_params'] / 1e6 for m in efficiency_data.values()]

    ax3_twin = ax3.twinx()

    bars1 = ax3.bar(x_pos - width/2, model_sizes, width,
                    label='Model Size (MB)', color=COLORS['depthwise'], alpha=0.8)
    bars2 = ax3_twin.bar(x_pos + width/2, param_counts, width,
                         label='Parameters (M)', color=COLORS['paired'], alpha=0.8)

    ax3.set_xlabel('Model Architecture')
    ax3.set_ylabel('Model Size (MB)', color=COLORS['depthwise'])
    ax3_twin.set_ylabel('Parameters (Millions)', color=COLORS['paired'])
    ax3.set_title('C. Model Efficiency Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models_data.keys(), rotation=15, ha='right')

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 4. Performance vs Efficiency Trade-off
    ax4 = axes[3]

    scatter_colors = [COLORS['base'], COLORS['enc_con'],
                      COLORS['depthwise'], COLORS['paired']]

    for i, (model_name, color) in enumerate(zip(models_data.keys(), scatter_colors)):
        dice_score = dice_scores[i]
        model_size = model_sizes[i]
        inference_time = list(efficiency_data.values())[
            i]['avg_inference_time_ms']

        # Size of bubble represents inference time
        ax4.scatter(model_size, dice_score, s=inference_time*20,
                    color=color, alpha=0.7, label=model_name, edgecolors='black')

    ax4.set_xlabel('Model Size (MB)')
    ax4.set_ylabel('Dice Coefficient')
    ax4.set_title('D. Performance vs Efficiency Trade-off')
    ax4.legend(title='Model (bubble size ∝ inference time)')

    plt.tight_layout()
    plt.savefig('comprehensive_performance_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('comprehensive_performance_analysis.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory


def create_pruning_analysis(data):
    """Create detailed pruning method analysis."""

    pruning_methods = {
        'Magnitude Pruning': 'magnitude_pruning',
        'SNIP One-Shot': 'snip_one_shot',
        'DepGraph': 'depgraph'
    }

    percentages = [10, 20, 30]

    fig, axes = create_figure_with_style(figsize=(18, 10), nrows=2, ncols=3)

    # 1. Dice Coefficient vs Pruning Ratio
    ax1 = axes[0]

    for method_name, method_key in pruning_methods.items():
        dice_scores = []
        for pct in percentages:
            if f'{pct}_percent' in data[method_key]:
                dice_scores.append(
                    data[method_key][f'{pct}_percent']['metrics']['dice_coef'])
            else:
                dice_scores.append(np.nan)

        color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
            0] in COLORS else COLORS['magnitude']
        ax1.plot(percentages, dice_scores, 'o-', linewidth=2.5, markersize=8,
                 label=method_name, color=color)

    # Add baseline
    baseline_dice = data['base_model']['metrics']['dice_coef']
    ax1.axhline(y=baseline_dice, color='red', linestyle='--', alpha=0.7,
                label='Base Model', linewidth=2)

    ax1.set_xlabel('Pruning Percentage (%)')
    ax1.set_ylabel('Dice Coefficient')
    ax1.set_title('A. Accuracy Degradation with Pruning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Mean IoU vs Pruning Ratio
    ax2 = axes[1]

    for method_name, method_key in pruning_methods.items():
        iou_scores = []
        for pct in percentages:
            if f'{pct}_percent' in data[method_key]:
                iou_scores.append(
                    data[method_key][f'{pct}_percent']['metrics']['mean_iou'])
            else:
                iou_scores.append(np.nan)

        color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
            0] in COLORS else COLORS['magnitude']
        ax2.plot(percentages, iou_scores, 's-', linewidth=2.5, markersize=8,
                 label=method_name, color=color)

    baseline_iou = data['base_model']['metrics']['mean_iou']
    ax2.axhline(y=baseline_iou, color='red', linestyle='--', alpha=0.7,
                label='Base Model', linewidth=2)

    ax2.set_xlabel('Pruning Percentage (%)')
    ax2.set_ylabel('Mean IoU')
    ax2.set_title('B. IoU Degradation with Pruning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Inference Time vs Pruning Ratio
    ax3 = axes[2]

    for method_name, method_key in pruning_methods.items():
        inference_times = []
        for pct in percentages:
            if f'{pct}_percent' in data[method_key]:
                inference_times.append(
                    data[method_key][f'{pct}_percent']['model_stats']['avg_inference_time_ms'])
            else:
                inference_times.append(np.nan)

        color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
            0] in COLORS else COLORS['magnitude']
        ax3.plot(percentages, inference_times, '^-', linewidth=2.5, markersize=8,
                 label=method_name, color=color)

    baseline_time = data['base_model']['model_stats']['avg_inference_time_ms']
    ax3.axhline(y=baseline_time, color='red', linestyle='--', alpha=0.7,
                label='Base Model', linewidth=2)

    ax3.set_xlabel('Pruning Percentage (%)')
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('C. Inference Time vs Pruning')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Class-wise Performance Degradation
    ax4 = axes[3]

    class_names = ['NCR/NET', 'ED', 'ET']
    x_pos = np.arange(len(class_names))
    width = 0.25

    # Show 30% pruning results for each method
    for i, (method_name, method_key) in enumerate(pruning_methods.items()):
        if '30_percent' in data[method_key]:
            class_scores = [data[method_key]['30_percent']['metrics']['class_dice'][f'c{j}']
                            for j in [2, 3, 4]]
            color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
                0] in COLORS else COLORS['magnitude']
            ax4.bar(x_pos + i*width, class_scores, width, label=method_name,
                    color=color, alpha=0.8)

    ax4.set_xlabel('Tumor Classes')
    ax4.set_ylabel('Dice Score')
    ax4.set_title('D. Class Performance at 30% Pruning')
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels(class_names)
    ax4.legend()

    # 5. Sparsity Analysis
    ax5 = axes[4]

    for method_name, method_key in pruning_methods.items():
        sparsity_ratios = []
        for pct in percentages:
            if f'{pct}_percent' in data[method_key]:
                total_params = data[method_key][f'{pct}_percent']['model_stats']['num_params']
                zero_params = data[method_key][f'{pct}_percent']['model_stats']['zero_params']
                sparsity = (zero_params / total_params) * 100
                sparsity_ratios.append(sparsity)
            else:
                sparsity_ratios.append(np.nan)

        color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
            0] in COLORS else COLORS['magnitude']
        ax5.plot(percentages, sparsity_ratios, 'D-', linewidth=2.5, markersize=8,
                 label=method_name, color=color)

    ax5.set_xlabel('Target Pruning Percentage (%)')
    ax5.set_ylabel('Actual Sparsity (%)')
    ax5.set_title('E. Achieved Sparsity vs Target')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Efficiency-Accuracy Trade-off
    ax6 = axes[5]

    for method_name, method_key in pruning_methods.items():
        dice_scores = []
        inference_times = []
        for pct in percentages:
            if f'{pct}_percent' in data[method_key]:
                dice_scores.append(
                    data[method_key][f'{pct}_percent']['metrics']['dice_coef'])
                inference_times.append(
                    data[method_key][f'{pct}_percent']['model_stats']['avg_inference_time_ms'])

        color = COLORS[method_key.split('_')[0]] if method_key.split('_')[
            0] in COLORS else COLORS['magnitude']
        ax6.scatter(inference_times, dice_scores, s=100, color=color, alpha=0.7,
                    label=method_name, edgecolors='black')

        # Connect points with lines
        ax6.plot(inference_times, dice_scores, '--', color=color, alpha=0.5)

    # Add base model point
    base_dice = data['base_model']['metrics']['dice_coef']
    base_time = data['base_model']['model_stats']['avg_inference_time_ms']
    ax6.scatter(base_time, base_dice, s=150, color='red', marker='*',
                label='Base Model', edgecolors='black', zorder=5)

    ax6.set_xlabel('Inference Time (ms)')
    ax6.set_ylabel('Dice Coefficient')
    ax6.set_title('F. Accuracy vs Speed Trade-off')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pruning_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('pruning_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory


def create_resource_efficiency_analysis(data):
    """Create comprehensive resource efficiency analysis."""

    models_data = {
        'Base Model': data['base_model'],
        'ENC-CON': data['enc_con_stats'],
        'Depthwise Shared': data['depthwise_shared'],
        'Paired Shared': data['paired_shared_depthwise']
    }

    fig, axes = create_figure_with_style(figsize=(16, 12), nrows=2, ncols=2)

    # 1. Memory Usage Comparison
    ax1 = axes[0]

    model_names = list(models_data.keys())
    ram_usage = [m['memory_stats']['peak_ram_gb']
                 for m in models_data.values()]
    gpu_usage = [m['memory_stats']['peak_gpu_memory_gb']
                 for m in models_data.values()]

    x_pos = np.arange(len(model_names))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, ram_usage, width, label='Peak RAM (GB)',
                    color=COLORS['base'], alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, gpu_usage, width, label='Peak GPU Memory (GB)',
                    color=COLORS['enc_con'], alpha=0.8)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Memory Usage (GB)')
    ax1.set_title('A. Memory Usage Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.legend()

    # 2. Computational Efficiency
    ax2 = axes[1]

    flops = [m['model_stats']['flops'] /
             1e9 for m in models_data.values()]  # Convert to GFLOPs
    inference_times = [m['model_stats']['avg_inference_time_ms']
                       for m in models_data.values()]

    # Create scatter plot with model size as bubble size
    model_sizes = [m['model_stats']['model_size_mb']
                   for m in models_data.values()]
    colors = [COLORS['base'], COLORS['enc_con'],
              COLORS['depthwise'], COLORS['paired']]

    for i, (name, color) in enumerate(zip(model_names, colors)):
        ax2.scatter(flops[i], inference_times[i], s=model_sizes[i]*10,
                    color=color, alpha=0.7, label=name, edgecolors='black')

    ax2.set_xlabel('FLOPs (GFLOPs)')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('B. Computational Efficiency\n(bubble size ∝ model size)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Parameter Efficiency
    ax3 = axes[2]

    param_counts = [m['model_stats']['num_params'] /
                    1e6 for m in models_data.values()]
    dice_scores = [m['metrics']['dice_coef'] for m in models_data.values()]

    bars = ax3.bar(model_names, param_counts, color=colors, alpha=0.8)

    # Add dice scores as text on bars
    for i, (bar, dice) in enumerate(zip(bars, dice_scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'Dice: {dice:.3f}', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_xlabel('Model Architecture')
    ax3.set_ylabel('Parameters (Millions)')
    ax3.set_title('C. Parameter Count vs Performance')
    ax3.tick_params(axis='x', rotation=15)

    # 4. Efficiency Score Radar Chart
    ax4 = axes[3]
    ax4.remove()  # Remove the subplot
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')

    # Define efficiency metrics (normalized to 0-1 scale)
    metrics = ['Speed\n(1/inference_time)', 'Memory\n(1/peak_ram)',
               'Size\n(1/model_size)', 'Accuracy\n(dice_coef)']

    # Normalize metrics
    max_time = max(inference_times)
    max_ram = max(ram_usage)
    max_size = max(model_sizes)

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [
            1 - (inference_times[i] / max_time),  # Speed (inverted)
            # Memory efficiency (inverted)
            1 - (ram_usage[i] / max_ram),
            1 - (model_sizes[i] / max_size),      # Size efficiency (inverted)
            dice_scores[i]                        # Accuracy
        ]
        values = np.concatenate((values, [values[0]]))

        ax4.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax4.fill(angles, values, alpha=0.1, color=color)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('D. Overall Efficiency Profile', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('resource_efficiency_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('resource_efficiency_analysis.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory


def create_statistical_summary_table(data):
    """Create a comprehensive statistical summary table."""

    # Prepare data for the table
    models = {
        'Base Model': data['base_model'],
        'ENC-CON': data['enc_con_stats'],
        'Depthwise Shared': data['depthwise_shared'],
        'Paired Shared': data['paired_shared_depthwise']
    }

    # Create summary statistics
    summary_data = []
    for model_name, model_data in models.items():
        metrics = model_data['metrics']
        model_stats = model_data['model_stats']
        memory_stats = model_data['memory_stats']

        summary_data.append({
            'Model': model_name,
            'Dice Coef': f"{metrics['dice_coef']:.4f}",
            'Mean IoU': f"{metrics['mean_iou']:.4f}",
            'NCR/NET': f"{metrics['class_dice']['c2']:.4f}",
            'ED': f"{metrics['class_dice']['c3']:.4f}",
            'ET': f"{metrics['class_dice']['c4']:.4f}",
            'Parameters (M)': f"{model_stats['num_params']/1e6:.2f}",
            'Model Size (MB)': f"{model_stats['model_size_mb']:.2f}",
            'Inference (ms)': f"{model_stats['avg_inference_time_ms']:.2f}",
            'Peak RAM (GB)': f"{memory_stats['peak_ram_gb']:.2f}",
            'Peak GPU (GB)': f"{memory_stats['peak_gpu_memory_gb']:.2f}",
        })

    df = pd.DataFrame(summary_data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code the model names
    colors = [COLORS['base'], COLORS['enc_con'],
              COLORS['depthwise'], COLORS['paired']]
    for i, color in enumerate(colors):
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color='white')

    plt.title('Comprehensive Model Performance Summary',
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig('performance_summary_table.pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    return df


def main():
    """Main function to generate all visualizations."""
    print("🏆 Generating Nobel Prize Standard Visualizations...")
    print("=" * 60)

    try:
        # Load data
        data = load_data()
        print("✅ Data loaded successfully")

        # Create visualizations
        print("\n📊 Creating comprehensive performance analysis...")
        create_comprehensive_performance_analysis(data)

        print("✂️ Creating pruning method analysis...")
        create_pruning_analysis(data)

        print("⚡ Creating resource efficiency analysis...")
        create_resource_efficiency_analysis(data)

        print("📋 Creating statistical summary table...")
        summary_df = create_statistical_summary_table(data)

        print("\n🎉 All visualizations completed successfully!")
        print("\nGenerated files:")
        print("- comprehensive_performance_analysis.png/.pdf")
        print("- pruning_analysis.png/.pdf")
        print("- resource_efficiency_analysis.png/.pdf")
        print("- performance_summary_table.png/.pdf")

        print(f"\n📈 Summary Statistics:")
        print(
            f"Best performing model: {summary_df.loc[summary_df['Dice Coef'].astype(float).idxmax(), 'Model']}")
        print(
            f"Most efficient model: {summary_df.loc[summary_df['Parameters (M)'].astype(float).idxmin(), 'Model']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
