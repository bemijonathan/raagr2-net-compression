Lite Models Creation using Pruning and Sharing Weights on RAAGR2-Net Models


This dissertation investigates the application of model pruning and weight sharing techniques to the RAAGR2-Net architecture, aiming to reduce computational complexity and enhance deployment efficiency without compromising segmentation accuracy. The study will explore various pruning strategies—such as structured and unstructured pruning—and weight sharing methods to identify optimal configurations that maintain or improve the model's performance on brain tumor segmentation tasks. Evaluation metrics will include model size, inference speed, and segmentation accuracy on benchmark datasets like BraTS. The ultimate goal is to develop a streamlined version of RAAGR2-Net suitable for real-world clinical applications, particularly in resource-constrained environment



1. Introduction
- Rising computational demands of neural networks vs. deployment constraints in medical settings and edge deployment.
- Brain tumor segmentation as a computationally intensive medical imaging task
- Definition of RAAGR2 (Recurrent Attention Atrous Spatial Pyramid Pooling R2) models for segmentation (just describe very briefly.)
- Economic and practical benefits of model compression in medical imaging workflows
- Research gap: systematic evaluation of pruning and parameter sharing in specialized medical segmentation architectures

2. Literature Review and Related Works
- Neural network pruning approaches (add the classes, structured and Unstructured, Learning based.)
  - Magnitude-based (Han et al., 2015)
  - Network slimming (Liu et al., 2017) 
  - SNIP connection sensitivity (Lee et al., 2019)
  - Dependency graph pruning (BN-scale based approaches)
- Parameter sharing strategies
  - Convolutional filter sharing in CNNs
- U-Net derivatives for medical image segmentation
  - Attention U-Net, R2U-Net, ASPP enhanced models
- Position of RAAGR2: combines recurrent blocks, attention mechanisms, and ASPP

3. Methodology
- RAAGR2 Architecture
  - Base U-Net encoder-decoder structure
  - ReASPP3 module with dilated convolutions
  - Recurrent CNN blocks (RRCNNBlock) with residual connections
  - Attention mechanism for feature refinement
  - Dimensions and channel counts in each layer
  - Loss function: Weighted BCE + Dice loss




- Pruning Implementation
  - Magnitude-based pruning
    - L2 norm calculation of filter weights
    - Percentile-based thresholding for filter removal
    - Structured vs. unstructured pruning impacts
  
  - Network slimming
    - L1 regularization on batch normalization scaling factors
    - Channel importance determination via gamma parameters
    - Progressive channel removal and model restructuring
  
  - SNIP pruning
    - Connection sensitivity calculation using gradient information
    - One-shot pruning at initialization
    - Preservation of important connections based on learning signal
  
  - DeGraph pruning
    - Dependency graph construction for network layers
    - BN-scale importance metrics for pruning decisions
    - Handling of special cases (grouped convolutions, output layers)

- Shared-Weight Implementation
  - SharedDepthwiseBlock design
    - Shared 3×3 depthwise convolution kernel across multiple dilations
    - Individual pointwise convolutions for feature transformation
    - Residual projections for maintaining information flow
  
  - Integration with ReASPP3
    - Single shared kernel applied with varying dilation rates
    - Weight tensor shape: (in_channels, 1, 3, 3)
    - Parameter reduction analysis
    - Weight tying in transformers and recurrent architectures



5. Experiments & Results

4. Implementation Details
- Codebase organization
  - Architecture modules (model.py, shared_model.py)
  - Pruning techniques (magnitude_based_pruning.py, slimming.py, etc.)
  - Shared training implementation
  - Evaluation metrics and visualization

- Training setup
  - Optimizer: Adam (learning rate: 1e-4 to 1e-5)
  - Learning rate scheduling with ReduceLROnPlateau
  - Training epochs: 30-70 depending on technique

- Evaluation metrics
  - Dice coefficient (overall and per tumor region)
  - Mean IoU
  - Model size (parameter count)
  - Inference speed benchmarks

- Dataset: Brain tumor segmentation dataset
  - Multiple MRI modalities (T1, T1ce, T2, FLAIR)
  - Three segmentation targets (NCR/NET, ET, ED)
  - Training/validation/test split ratios

- Magnitude-based pruning results
  - Accuracy vs. pruning ratio trade-off curve
  - Impact on different tumor regions
  - Optimal pruning threshold determination

- Network slimming analysis
  - Effect of L1 regularization strength
  - Channel reduction at different network depths
  - Fine-tuning effectiveness post-pruning

- SNIP pruning performance
  - Single-shot vs. iterative pruning comparison
  - Preservation of critical network connections
  - Limitations in medical imaging context

- DeGraph pruning outcomes
  - Dependency-aware vs. naive pruning
  - Structural integrity maintenance
  - Handling of specialized architecture components

- Shared weights impact
  - Parameter reduction from weight sharing (30-40%)
  - Performance comparison with original model
  - Training convergence behavior

- Combined approach evaluation
  - Synergistic effects of pruning + weight sharing
  - Optimal technique combinations
  - Extreme compression scenarios

6. Discussion
- Trade-off analysis
  - Parameter efficiency vs. segmentation accuracy
  - Impact on specific tumor regions (which are more resilient to pruning)
  - Inference speed improvements on target hardware

- Architectural insights
  - Most important layers/filters for brain segmentation
  - Redundancy patterns in medical imaging networks
  - Over-parameterization in current architectures

- Limitations
  - Generalization to other medical imaging tasks
  - Hardware-specific optimization constraints
  - Manual hyperparameter tuning requirements

- Clinical relevance
  - Practical deployment considerations
  - Acceptable accuracy thresholds for clinical use
  - Resource constraints in medical settings

7. Conclusion & Future Work
- Key findings
  - Achievable compression rates with minimal accuracy loss
  - Optimal pruning techniques for RAAGR2 architecture
  - Effectiveness of shared depthwise convolutions

- Future directions
  - Automated pruning ratio selection
  - Task-specific pruning strategies
  - Dynamic pruning during inference
  - Mixed-precision integration
  - Distillation from pruned teachers
  - Extension to 3D medical volumes

8. References
- Relevant literature on neural network compression
- Medical image segmentation papers
- Brain tumor segmentation benchmark papers
- Original architecture papers (U-Net, ASPP, attention mechanisms)
- Pruning technique foundational papers


