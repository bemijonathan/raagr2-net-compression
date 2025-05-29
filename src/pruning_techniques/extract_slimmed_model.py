import os
import torch
import torch.nn as nn
import argparse
import inspect
from pprint import pprint
import json
from collections import OrderedDict

class ModelAnalyzer:
    def __init__(self, model):
        self.model = model
        self.structure = OrderedDict()
        self.module_counts = {}
    
    def analyze_model(self):
        """Analyze model structure to extract architecture details"""
        # Extract top-level modules
        for name, module in self.model.named_children():
            self.structure[name] = self._analyze_module(module)
        
        return self.structure
    
    def _analyze_module(self, module):
        """Recursively analyze a module to extract its structure"""
        if isinstance(module, nn.Conv2d):
            return {
                "type": "Conv2d",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "groups": module.groups,
                "bias": module.bias is not None
            }
        elif isinstance(module, nn.ConvTranspose2d):
            return {
                "type": "ConvTranspose2d",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "bias": module.bias is not None
            }
        elif isinstance(module, nn.BatchNorm2d):
            return {
                "type": "BatchNorm2d",
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum
            }
        elif isinstance(module, nn.MaxPool2d):
            return {
                "type": "MaxPool2d",
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding if hasattr(module, 'padding') else 0
            }
        elif isinstance(module, nn.Sequential):
            seq_layers = OrderedDict()
            for i, layer in enumerate(module):
                seq_layers[f"layer{i}"] = self._analyze_module(layer)
            return {
                "type": "Sequential",
                "layers": seq_layers
            }
        elif hasattr(module, "__class__") and module.__class__.__name__ == "ReASPP3":
            # Special handling for ReASPP3
            result = {
                "type": "ReASPP3",
                "in_channels": self._find_in_channels(module),
                "out_channels": self._find_out_channels(module),
                "r": self._find_dilation(module)
            }
            # Track module counts for naming
            self.module_counts["ReASPP3"] = self.module_counts.get("ReASPP3", 0) + 1
            return result
        elif hasattr(module, "__class__") and module.__class__.__name__ == "RRCNNBlock":
            # Handle RRCNNBlock
            result = {
                "type": "RRCNNBlock",
                "in_channels": self._find_in_channels(module),
                "out_channels": self._find_out_channels(module),
                "t": self._find_recurrent_steps(module)
            }
            self.module_counts["RRCNNBlock"] = self.module_counts.get("RRCNNBlock", 0) + 1
            return result
        elif hasattr(module, "__class__") and module.__class__.__name__ == "AttentionBlock":
            # Handle AttentionBlock
            result = {
                "type": "AttentionBlock",
                "F_g": self._find_attention_channels(module, "g"),
                "F_l": self._find_attention_channels(module, "l"),
                "F_int": self._find_attention_channels(module, "int")
            }
            self.module_counts["AttentionBlock"] = self.module_counts.get("AttentionBlock", 0) + 1
            return result
        elif hasattr(module, "__class__") and module.__class__.__name__ == "UpConv":
            # Handle UpConv
            result = {
                "type": "UpConv",
                "in_channels": self._find_in_channels(module),
                "out_channels": self._find_out_channels(module)
            }
            self.module_counts["UpConv"] = self.module_counts.get("UpConv", 0) + 1
            return result
        else:
            # Handle other module types by recursively analyzing children
            children = OrderedDict()
            for name, child in module.named_children():
                children[name] = self._analyze_module(child)
            
            if not children:
                # No children, return basic info
                return {
                    "type": module.__class__.__name__,
                    "params": str(module)
                }
            else:
                return {
                    "type": module.__class__.__name__,
                    "children": children
                }
    
    def _find_in_channels(self, module):
        """Find input channels for a module"""
        # First try direct attribute
        if hasattr(module, "in_channels"):
            return module.in_channels
        
        # Look for first Conv2d or ConvTranspose2d
        for _, child in module.named_modules():
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                return child.in_channels
        
        # Fallback - inspect parameters
        for name, child in module.named_children():
            if "conv" in name.lower() and hasattr(child, "in_channels"):
                return child.in_channels
        
        return None
    
    def _find_out_channels(self, module):
        """Find output channels for a module"""
        # First try direct attribute
        if hasattr(module, "out_channels"):
            return module.out_channels
        
        # Look for last Conv2d or BatchNorm2d
        last_conv = None
        last_bn = None
        
        for _, child in module.named_modules():
            if isinstance(child, nn.Conv2d):
                last_conv = child
            elif isinstance(child, nn.BatchNorm2d):
                last_bn = child
        
        if last_bn:
            return last_bn.num_features
        elif last_conv:
            return last_conv.out_channels
        
        return None
    
    def _find_dilation(self, module):
        """Find dilation rate for ReASPP3"""
        for _, child in module.named_modules():
            if isinstance(child, nn.Conv2d) and child.dilation[0] > 1:
                return child.dilation[0] // 2  # ReASPP3 uses r*2, r*3 so we divide
        return 3  # Default value
    
    def _find_recurrent_steps(self, module):
        """Find recurrent steps (t) in RecurrentBlock or RRCNNBlock"""
        # Look for RecurrentBlock
        for _, child in module.named_modules():
            if hasattr(child, "__class__") and child.__class__.__name__ == "RecurrentBlock":
                if hasattr(child, "t"):
                    return child.t
        return 2  # Default value
    
    def _find_attention_channels(self, module, channel_type):
        """Find channel dimensions for AttentionBlock"""
        if channel_type == "g" and hasattr(module, "W_g"):
            for layer in module.W_g:
                if isinstance(layer, nn.Conv2d):
                    return layer.in_channels
        elif channel_type == "l" and hasattr(module, "W_x"):
            for layer in module.W_x:
                if isinstance(layer, nn.Conv2d):
                    return layer.in_channels
        elif channel_type == "int" and hasattr(module, "W_g"):
            for layer in module.W_g:
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
        
        return None


class ModelCodeGenerator:
    def __init__(self, structure):
        self.structure = structure
        
    def generate_model_code(self):
        """Generate Python code for recreating the model"""
        code = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "# Basic building blocks",
            self._generate_building_blocks(),
            "",
            "class SlimmedDLUNet(nn.Module):",
            "    def __init__(self, in_channels=4, out_channels=5):",
            "        super(SlimmedDLUNet, self).__init__()",
            "",
            "        # Encoder path"
        ]
        
        # Add encoder components
        enc_layers = [k for k in self.structure.keys() if k.startswith('enc')]
        pool_layers = [k for k in self.structure.keys() if k.startswith('pool')]
        
        for enc in enc_layers:
            params = self._format_params(self.structure[enc])
            code.append(f"        self.{enc} = ReASPP3{params}")
        
        for pool in pool_layers:
            params = self._format_params(self.structure[pool])
            code.append(f"        self.{pool} = nn.MaxPool2d{params}")
        
        code.append("")
        code.append("        # Decoder path")
        
        # Add decoder components
        up_layers = [k for k in self.structure.keys() if k.startswith('up')]
        att_layers = [k for k in self.structure.keys() if k.startswith('att')]
        dec_layers = [k for k in self.structure.keys() if k.startswith('dec')]
        
        for up in up_layers:
            params = self._format_params(self.structure[up])
            code.append(f"        self.{up} = UpConv{params}")
        
        for att in att_layers:
            params = self._format_params(self.structure[att])
            code.append(f"        self.{att} = AttentionBlock{params}")
        
        for dec in dec_layers:
            params = self._format_params(self.structure[dec])
            code.append(f"        self.{dec} = RRCNNBlock{params}")
        
        # Add final layers
        if 'final_conv' in self.structure:
            params = self._format_params(self.structure['final_conv'])
            code.append(f"        self.final_conv = nn.Conv2d{params}")
        
        if 'sigmoid' in self.structure:
            code.append("        self.sigmoid = nn.Sigmoid()")
        
        # Add forward method
        code.extend([
            "",
            "    def forward(self, x):",
            "        # Encoder path",
            "        e1 = self.enc1(x)",
            "        p1 = self.pool1(e1)",
            "",
            "        e2 = self.enc2(p1)",
            "        p2 = self.pool2(e2)",
            "",
            "        e3 = self.enc3(p2)",
            "        p3 = self.pool3(e3)",
            "",
            "        e4 = self.enc4(p3)",
            "        p4 = self.pool4(e4)",
            "",
            "        e5 = self.enc5(p4)",
            "",
            "        # Decoder path with attention and skip connections",
            "        d5 = self.up5(e5)",
            "        a4 = self.att5(d5, e4)",
            "        d5 = torch.cat([a4, d5], dim=1)",
            "        d5 = self.dec5(d5)",
            "",
            "        d4 = self.up4(d5)",
            "        a3 = self.att4(d4, e3)",
            "        d4 = torch.cat([a3, d4], dim=1)",
            "        d4 = self.dec4(d4)",
            "",
            "        d3 = self.up3(d4)",
            "        a2 = self.att3(d3, e2)",
            "        d3 = torch.cat([a2, d3], dim=1)",
            "        d3 = self.dec3(d3)",
            "",
            "        d2 = self.up2(d3)",
            "        a1 = self.att2(d2, e1)",
            "        d2 = torch.cat([a1, d2], dim=1)",
            "        d2 = self.dec2(d2)",
            "",
            "        out = self.final_conv(d2)",
            "        return self.sigmoid(out)"
        ])
        
        return "\n".join(code)
    
    def _generate_building_blocks(self):
        """Generate code for the building block classes"""
        blocks = []
        
        # Basic blocks needed for DLUNet
        blocks.extend([
            "class ConvBlock(nn.Module):",
            "    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):",
            "        super(ConvBlock, self).__init__()",
            "        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)",
            "        self.bn = nn.BatchNorm2d(out_channels)",
            "        self.relu = nn.ReLU(inplace=True)",
            "",
            "    def forward(self, x):",
            "        return self.relu(self.bn(self.conv(x)))",
            "",
            "class ConvOne(nn.Module):",
            "    def __init__(self, in_channels, out_channels):",
            "        super(ConvOne, self).__init__()",
            "        self.conv = nn.Sequential(",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1),",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "    def forward(self, x):",
            "        return self.conv(x)",
            "",
            "class RecurrentBlock(nn.Module):",
            "    def __init__(self, in_channels, out_channels, t=2):",
            "        super(RecurrentBlock, self).__init__()",
            "        self.t = t",
            "        self.conv = ConvOne(in_channels, out_channels)",
            "",
            "    def forward(self, x):",
            "        x1 = x",
            "        for i in range(self.t):",
            "            if i == 0:",
            "                x1 = self.conv(x)",
            "            x1 = x + x1",
            "            x1 = self.conv(x1)",
            "        return x1",
            "",
            "class RRCNNBlock(nn.Module):",
            "    def __init__(self, in_channels, out_channels, t=2):",
            "        super(RRCNNBlock, self).__init__()",
            "        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)",
            "        self.dropout = nn.Dropout(0.2)",
            "        self.recblock1 = RecurrentBlock(out_channels, out_channels, t)",
            "        self.recblock2 = RecurrentBlock(out_channels, out_channels, t)",
            "",
            "    def forward(self, x):",
            "        x = self.conv1x1(x)",
            "        x = self.dropout(x)",
            "        x1 = self.recblock1(x)",
            "        x1 = self.recblock2(x1)",
            "        return x + x1",
            "",
            "class UpConv(nn.Module):",
            "    def __init__(self, in_channels, out_channels):",
            "        super(UpConv, self).__init__()",
            "        self.up = nn.Sequential(",
            "            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "    def forward(self, x):",
            "        return self.up(x)",
            "",
            "class AttentionBlock(nn.Module):",
            "    def __init__(self, F_g, F_l, F_int):",
            "        super(AttentionBlock, self).__init__()",
            "        self.W_g = nn.Sequential(",
            "            nn.Conv2d(F_g, F_int, kernel_size=1),",
            "            nn.BatchNorm2d(F_int)",
            "        )",
            "",
            "        self.W_x = nn.Sequential(",
            "            nn.Conv2d(F_l, F_int, kernel_size=1),",
            "            nn.BatchNorm2d(F_int)",
            "        )",
            "",
            "        self.psi = nn.Sequential(",
            "            nn.Conv2d(F_int, 1, kernel_size=1),",
            "            nn.BatchNorm2d(1),",
            "            nn.Sigmoid()",
            "        )",
            "",
            "        self.relu = nn.ReLU(inplace=True)",
            "",
            "    def forward(self, g, x):",
            "        g1 = self.W_g(g)",
            "        x1 = self.W_x(x)",
            "        psi = self.relu(g1 + x1)",
            "        psi = self.psi(psi)",
            "",
            "        return x * psi",
            "",
            "class ReASPP3(nn.Module):",
            "    def __init__(self, in_channels, out_channels, r=3):",
            "        super(ReASPP3, self).__init__()",
            "",
            "        # Calculate groups to ensure compatibility",
            "        groups = 1",
            "",
            "        # Standard convolution path",
            "        self.conv1_1 = nn.Sequential(",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups),",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        self.conv1_2 = nn.Sequential(",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        # Dilated conv path 1",
            "        self.conv2_1 = nn.Sequential(",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, groups=groups),",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        self.conv2_2 = nn.Sequential(",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        # Dilated conv path 2",
            "        self.conv3_1 = nn.Sequential(",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r*2, dilation=r*2, groups=groups),",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        self.conv3_2 = nn.Sequential(",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        # Dilated conv path 3",
            "        self.conv4_1 = nn.Sequential(",
            "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r*3, dilation=r*3, groups=groups),",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        self.conv4_2 = nn.Sequential(",
            "            nn.Conv2d(out_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "        # Final 1x1 conv to reduce channels",
            "        self.final_conv = nn.Sequential(",
            "            nn.Conv2d(out_channels*4 + in_channels, out_channels, kernel_size=1),",
            "            nn.BatchNorm2d(out_channels),",
            "            nn.ReLU(inplace=True)",
            "        )",
            "",
            "    def forward(self, x):",
            "        # Path 1",
            "        x1_1 = self.conv1_1(x)",
            "        x1_2 = self.conv1_2(x1_1)",
            "        x1_2 = x1_2 + x1_1",
            "",
            "        # Path 2",
            "        x2_1 = self.conv2_1(x)",
            "        x2_2 = self.conv2_2(x2_1)",
            "        x2_2 = x2_2 + x2_1",
            "",
            "        # Path 3",
            "        x3_1 = self.conv3_1(x)",
            "        x3_2 = self.conv3_2(x3_1)",
            "        x3_2 = x3_2 + x3_1",
            "",
            "        # Path 4",
            "        x4_1 = self.conv4_1(x)",
            "        x4_2 = self.conv4_2(x4_1)",
            "        x4_2 = x4_2 + x4_1",
            "",
            "        # Concatenate all paths with input",
            "        x_cat = torch.cat([x1_2, x2_2, x3_2, x4_2, x], dim=1)",
            "",
            "        # Final 1x1 conv",
            "        out = self.final_conv(x_cat)",
            "",
            "        return out"
        ])
        
        return "\n".join(blocks)
    
    def _format_params(self, module_info):
        """Format parameters for code generation"""
        if module_info["type"] == "ReASPP3":
            params = f"({module_info['in_channels']}, {module_info['out_channels']}, {module_info.get('r', 3)})"
        elif module_info["type"] == "RRCNNBlock":
            params = f"({module_info['in_channels']}, {module_info['out_channels']}, {module_info.get('t', 2)})"
        elif module_info["type"] == "AttentionBlock":
            params = f"({module_info['F_g']}, {module_info['F_l']}, {module_info['F_int']})"
        elif module_info["type"] == "UpConv":
            params = f"({module_info['in_channels']}, {module_info['out_channels']})"
        elif module_info["type"] == "Conv2d":
            params = f"({module_info['in_channels']}, {module_info['out_channels']}, {module_info['kernel_size']})"
        elif module_info["type"] == "MaxPool2d":
            params = f"(kernel_size={module_info['kernel_size']}, stride={module_info['stride']})"
        else:
            # Generic parameter formatting
            params = "(...)"
        
        return params


class SlimmedModelCreator:
    @staticmethod
    def create_model_from_code(code_string, model_name="SlimmedDLUNet"):
        """Dynamically create model from generated code"""
        # Create a temporary module namespace
        module_namespace = {}
        
        # Execute the code in this namespace
        exec(code_string, {"torch": torch, "nn": nn, "F": torch.nn.functional}, module_namespace)
        
        # Get the model class
        model_class = module_namespace.get(model_name)
        
        if model_class is None:
            raise ValueError(f"Model class {model_name} not found in generated code")
        
        # Create an instance of the model
        model = model_class()
        
        return model
    
    @staticmethod
    def transfer_weights(source_model, target_model):
        """Transfer weights from source model to target model where architectures match"""
        # This is a complex operation that would ideally match layers
        # based on their position and dimensions, but for simplicity
        # we'll just copy weights for layers with matching names and shapes
        
        print("Transferring weights from source to target model...")
        
        # Get state dictionaries
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        # Create new state dict for target
        new_target_dict = target_dict.copy()
        
        # Track transferred parameters
        transferred = 0
        total_target = 0
        
        for target_key in target_dict.keys():
            total_target += 1
            
            # Find matching source key
            source_key = target_key  # Direct match
            
            # If source has this key and dimensions match
            if source_key in source_dict and source_dict[source_key].shape == target_dict[target_key].shape:
                new_target_dict[target_key] = source_dict[source_key]
                transferred += 1
        
        # Load the new state dict
        target_model.load_state_dict(new_target_dict)
        
        print(f"Transferred {transferred}/{total_target} parameters")
        
        return target_model


def extract_and_save_model(slimmed_model_path, output_dir='model/reconstructed'):
    """Extract slimmed model architecture and save a clean implementation"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load slimmed model
    print(f"Loading slimmed model from {slimmed_model_path}")
    slimmed_model = torch.load(slimmed_model_path, map_location=torch.device('cpu'))
    
    # Analyze model structure
    print("Analyzing model structure...")
    analyzer = ModelAnalyzer(slimmed_model)
    structure = analyzer.analyze_model()
    
    # Save structure as JSON for reference
    structure_path = os.path.join(output_dir, 'model_structure.json')
    with open(structure_path, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"Model structure saved to {structure_path}")
    
    # Generate Python code for the model
    print("Generating model code...")
    code_generator = ModelCodeGenerator(structure)
    model_code = code_generator.generate_model_code()
    
    # Save model code
    code_path = os.path.join(output_dir, 'slimmed_model.py')
    with open(code_path, 'w') as f:
        f.write(model_code)
    print(f"Model code saved to {code_path}")
    
    # Create new model from generated code
    print("Creating clean model from generated code...")
    model_creator = SlimmedModelCreator()
    new_model = model_creator.create_model_from_code(model_code)
    
    # Transfer weights where possible
    print("Transferring weights from slimmed model...")
    new_model = model_creator.transfer_weights(slimmed_model, new_model)
    
    # Save the new model
    new_model_path = os.path.join(output_dir, 'slimmed_model_clean.pth')
    torch.save(new_model, new_model_path)
    print(f"Clean model saved to {new_model_path}")
    
    # Print model summary
    param_count = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total Parameters: {param_count:,}")
    
    return {
        'structure_path': structure_path,
        'code_path': code_path,
        'model_path': new_model_path,
        'parameter_count': param_count
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and recreate slimmed model")
    parser.add_argument("--model_path", required=True, 
                        help="Path to the slimmed model (.pth file)")
    parser.add_argument("--output_dir", default="model/reconstructed",
                        help="Directory to save reconstructed model files")
    
    args = parser.parse_args()
    
    results = extract_and_save_model(args.model_path, args.output_dir)
    
    print("\nProcess completed successfully!")
    print(f"Generated model has {results['parameter_count']:,} parameters")
    print("You can now use the clean model implementation from:")
    print(f"  Code: {results['code_path']}")
    print(f"  Model weights: {results['model_path']}")