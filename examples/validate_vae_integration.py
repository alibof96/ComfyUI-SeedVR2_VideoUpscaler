#!/usr/bin/env python3
"""
Test script for VAE integration
Tests the complete VAE integration pipeline without requiring actual model files
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_registry():
    """Test model registry has correct VAE entries"""
    print("=" * 60)
    print("Test 1: Model Registry")
    print("=" * 60)
    
    from src.utils.model_registry import (
        MODEL_REGISTRY, 
        DEFAULT_VAE, 
        get_available_vaes,
        get_default_vaes
    )
    
    # Check taew2_1 is registered
    assert "taew2_1.safetensors" in MODEL_REGISTRY, "taew2_1.safetensors not in registry"
    print("✅ taew2_1.safetensors is registered")
    
    # Check it's categorized as VAE
    info = MODEL_REGISTRY["taew2_1.safetensors"]
    assert info.category == "vae", f"Expected category 'vae', got '{info.category}'"
    print(f"✅ Correctly categorized as VAE")
    
    # Check repo
    assert info.repo == "lightx2v/Autoencoders", f"Expected repo 'lightx2v/Autoencoders', got '{info.repo}'"
    print(f"✅ Correct repository: {info.repo}")
    
    # Check default VAE
    assert DEFAULT_VAE == "ema_vae_fp16.safetensors", "Default VAE changed unexpectedly"
    print(f"✅ Default VAE: {DEFAULT_VAE}")
    
    # Check available VAEs
    available_vaes = get_available_vaes()
    assert "ema_vae_fp16.safetensors" in available_vaes, "Default VAE not in available list"
    assert "taew2_1.safetensors" in available_vaes, "taew2_1 not in available list"
    print(f"✅ Available VAEs: {available_vaes}")
    
    print("\n")
    return True


def test_vae_compatibility_module():
    """Test VAE compatibility validation module"""
    print("=" * 60)
    print("Test 2: VAE Compatibility Module")
    print("=" * 60)
    
    from src.utils.vae_compatibility import validate_vae_compatibility, get_vae_info
    
    # Check functions are importable and callable
    assert callable(validate_vae_compatibility), "validate_vae_compatibility not callable"
    print("✅ validate_vae_compatibility function available")
    
    assert callable(get_vae_info), "get_vae_info not callable"
    print("✅ get_vae_info function available")
    
    # Test with non-existent file (should handle gracefully)
    is_compatible, msg = validate_vae_compatibility("/nonexistent/file.safetensors")
    assert not is_compatible, "Should return False for non-existent file"
    assert "error" in msg.lower(), "Should return error message"
    print("✅ Handles non-existent files gracefully")
    
    # Test get_vae_info with non-existent file
    info = get_vae_info("/nonexistent/file.safetensors")
    assert info['error'] is not None, "Should set error field"
    assert not info['compatible'], "Should mark as not compatible"
    print("✅ get_vae_info handles errors correctly")
    
    print("\n")
    return True


def test_comfyui_node_interface():
    """Test ComfyUI node interface integration"""
    print("=" * 60)
    print("Test 3: ComfyUI Node Interface")
    print("=" * 60)
    
    from src.interfaces.comfyui_node import SeedVR2ExtraArgs
    
    # Test INPUT_TYPES has vae_model
    input_types = SeedVR2ExtraArgs.INPUT_TYPES()
    assert 'vae_model' in input_types['required'], "vae_model not in required inputs"
    print("✅ vae_model parameter in INPUT_TYPES")
    
    # Check vae_model options include both VAEs
    vae_options, vae_config = input_types['required']['vae_model']
    assert isinstance(vae_options, list), "VAE options should be a list"
    assert "ema_vae_fp16.safetensors" in vae_options, "Default VAE not in options"
    assert "taew2_1.safetensors" in vae_options, "taew2_1 not in options"
    print(f"✅ VAE options include both models: {len(vae_options)} total")
    
    # Check default
    assert vae_config['default'] == "ema_vae_fp16.safetensors", "Wrong default VAE"
    print(f"✅ Default VAE: {vae_config['default']}")
    
    # Test create_config includes vae_model
    import inspect
    sig = inspect.signature(SeedVR2ExtraArgs.create_config)
    params = list(sig.parameters.keys())
    assert 'vae_model' in params, "vae_model not in create_config parameters"
    print("✅ create_config accepts vae_model parameter")
    
    print("\n")
    return True


def test_function_signatures():
    """Test that all function signatures are updated correctly"""
    print("=" * 60)
    print("Test 4: Function Signatures")
    print("=" * 60)
    
    import inspect
    from src.core.generation import prepare_runner
    from src.core.model_manager import configure_runner
    
    # Test prepare_runner
    sig = inspect.signature(prepare_runner)
    params = list(sig.parameters.keys())
    assert 'vae_model' in params, "vae_model not in prepare_runner parameters"
    print("✅ prepare_runner has vae_model parameter")
    
    # Test configure_runner
    sig = inspect.signature(configure_runner)
    params = list(sig.parameters.keys())
    assert 'vae_model' in params, "vae_model not in configure_runner parameters"
    print("✅ configure_runner has vae_model parameter")
    
    print("\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           VAE Integration Test Suite                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n")
    
    tests = [
        ("Model Registry", test_model_registry),
        ("VAE Compatibility", test_vae_compatibility_module),
        ("ComfyUI Interface", test_comfyui_node_interface),
        ("Function Signatures", test_function_signatures),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! VAE integration is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
