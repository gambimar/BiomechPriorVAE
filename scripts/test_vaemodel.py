"""
Test script for VAEModelWrapper to verify model loading, inference, and gradient computation.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from vaemodel import VAEModelWrapper, initialize_vae, reconstruct, reconstruct_withgrad

def test_vae_model_wrapper():
    """Test the VAEModelWrapper class directly"""
    print("=" * 80)
    print("Testing VAEModelWrapper")
    print("=" * 80)
    
    # Paths to model and scaler
    model_path = "result/model/BiomechPriorVAE_best.pth"
    scaler_path = "result/model/scaler.pkl"
    
    # Model configuration
    num_dofs = 54  # 27 pos + 27 vel
    latent_dim = 24
    hidden_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Configuration: num_dofs={num_dofs}, latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    
    try:
        # Initialize the wrapper
        print("\n1. Loading model and scaler...")
        wrapper = VAEModelWrapper(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        print("✓ Model and scaler loaded successfully!")
        
        # Test with random input (27-dimensional for positions only)
        print("\n2. Testing reconstruction with 54-dimensional input (pos + vel)...")
        test_input_54 = np.random.randn(54).astype(np.float32) * 0.5
        print(f"Input shape: {test_input_54.shape}")
        print(f"Input range: [{test_input_54.min():.4f}, {test_input_54.max():.4f}]")
        
        # Test reconstruction without gradient
        print("\n3. Testing reconstruction (no gradient)...")
        error = wrapper.reconstruct(test_input_54, getgrad=False)
        print(f"✓ Reconstruction error: {error:.6f}")
        
        # Test reconstruction with gradient
        print("\n4. Testing reconstruction with gradient...")
        test_input_grad = test_input_54.copy()
        gradient = wrapper.reconstruct_withgrad(test_input_grad)
        print(f"✓ Gradient computed successfully!")
        print(f"Gradient shape: {gradient.shape}")
        print(f"Gradient range: [{gradient.min():.4f}, {gradient.max():.4f}]")
        print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
        
        # Test with zero velocities (still 54D but zeros in velocity part)
        print("\n5. Testing with zero velocities (54D with vel=0)...")
        test_input_zero_vel = np.zeros(54, dtype=np.float32)
        test_input_zero_vel[:27] = np.random.randn(27).astype(np.float32) * 0.5  # Random positions
        error_zero_vel = wrapper.reconstruct(test_input_zero_vel, getgrad=False)
        print(f"✓ Reconstruction error (zero vel): {error_zero_vel:.6f}")
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_global_functions():
    """Test the global initialize_vae, reconstruct, and reconstruct_withgrad functions"""
    print("\n" + "=" * 80)
    print("Testing Global Functions")
    print("=" * 80)
    
    # Paths to model and scaler
    model_path = "result/model/BiomechPriorVAE_best.pth"
    scaler_path = "result/model/scaler.pkl"
    
    # Model configuration
    num_dofs = 54
    latent_dim = 24
    hidden_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize
        print("\n1. Initializing VAE with global function...")
        success = initialize_vae(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        if not success:
            print("✗ Failed to initialize VAE")
            return False
        print("✓ VAE initialized successfully!")
        
        # Test reconstruction
        print("\n2. Testing global reconstruct function...")
        test_input = np.random.randn(54).astype(np.float32) * 0.5
        error = reconstruct(test_input)
        print(f"✓ Reconstruction error: {error:.6f}")
        
        # Test reconstruction with gradient
        print("\n3. Testing global reconstruct_withgrad function...")
        gradient = reconstruct_withgrad(test_input)
        print(f"✓ Gradient computed successfully!")
        print(f"Gradient shape: {gradient.shape}")
        print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
        
        print("\n" + "=" * 80)
        print("All global function tests passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_batch_processing():
    """Test processing multiple samples"""
    print("\n" + "=" * 80)
    print("Testing Batch Processing")
    print("=" * 80)
    
    model_path = "result/model/BiomechPriorVAE_best.pth"
    scaler_path = "result/model/scaler.pkl"
    num_dofs = 54
    latent_dim = 24
    hidden_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        wrapper = VAEModelWrapper(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        print("\n1. Processing batch of samples...")
        batch_size = 10
        errors = []
        
        for i in range(batch_size):
            test_input = np.random.randn(54).astype(np.float32) * 0.5
            error = wrapper.reconstruct(test_input, getgrad=False)
            errors.append(error)
        
        errors = np.array(errors)
        print(f"✓ Processed {batch_size} samples")
        print(f"Mean error: {errors.mean():.6f}")
        print(f"Std error: {errors.std():.6f}")
        print(f"Min error: {errors.min():.6f}")
        print(f"Max error: {errors.max():.6f}")
        
        print("\n" + "=" * 80)
        print("Batch processing test passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "VAE Model Testing Suite" + " " * 35 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_vae_model_wrapper()
    all_passed &= test_global_functions()
    all_passed &= test_batch_processing()
    
    # Summary
    print("\n\n")
    print("╔" + "═" * 78 + "╗")
    if all_passed:
        print("║" + " " * 25 + "ALL TESTS PASSED! ✓" + " " * 34 + "║")
    else:
        print("║" + " " * 25 + "SOME TESTS FAILED ✗" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    sys.exit(0 if all_passed else 1)
