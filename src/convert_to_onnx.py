import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import argparse

sys.path.append('models')
from pytorch_model import load_model


def convert_to_onnx(model_path="models/pytorch_model_weights.pth", 
                   output_path="model_artifacts/classification_model.onnx"):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model weights
        output_path: Output path for ONNX model
        
    Returns:
        bool: True if conversion successful
    """
    try:
        print(f"Loading PyTorch model from {model_path}")
        pytorch_model = load_model(model_path)
        pytorch_model.eval()
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create dummy input for conversion
        dummy_input = torch.randn(1, 3, 224, 224)
        
        print(f"Converting to ONNX: {output_path}")
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify conversion
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test conversion accuracy
        ort_session = ort.InferenceSession(output_path)
        
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)
        
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        max_diff = np.abs(pytorch_output.numpy() - onnx_output).max()
        
        if max_diff < 1e-5:
            print(f"Conversion successful! Max difference: {max_diff:.8f}")
            return True
        else:
            print(f"Conversion failed! Max difference: {max_diff:.8f}")
            return False
            
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model_path', default='models/pytorch_model_weights.pth',
                       help='Path to PyTorch model weights')
    parser.add_argument('--output_path', default='model_artifacts/classification_model.onnx',
                       help='Output path for ONNX model')
    
    args = parser.parse_args()
    
    success = convert_to_onnx(args.model_path, args.output_path)
    
    if success:
        print(f"ONNX model ready: {args.output_path}")
    else:
        print("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
