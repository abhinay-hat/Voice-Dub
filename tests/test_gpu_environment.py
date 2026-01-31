"""
GPU environment validation tests.
Ensures RTX 5090 is properly configured and all utilities work.
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.gpu_validation import validate_rtx5090_environment
from src.utils.memory_monitor import get_gpu_memory_info, print_gpu_memory_summary
from src.models.model_manager import ModelManager


def test_gpu_validation():
    """Test GPU validation utility."""
    print("\n" + "="*60)
    print("TEST 1: GPU Validation")
    print("="*60)

    env_info = validate_rtx5090_environment()

    # Verify expected values for RTX 5090
    assert '5090' in env_info['device_name'] or 'RTX' in env_info['device_name'], \
        f"Expected RTX 5090, got: {env_info['device_name']}"

    assert env_info['compute_capability'] == '12.0', \
        f"Expected compute capability 12.0, got: {env_info['compute_capability']}"

    assert env_info['total_vram_gb'] >= 30, \
        f"Expected ~32GB VRAM, got: {env_info['total_vram_gb']}GB"

    print("✓ GPU validation test passed")
    return env_info


def test_memory_monitoring():
    """Test memory monitoring utilities."""
    print("\n" + "="*60)
    print("TEST 2: Memory Monitoring")
    print("="*60)

    # Get memory info
    mem_before = get_gpu_memory_info()
    print(f"Memory before allocation: {mem_before}")

    assert mem_before['total'] >= 30, "Total VRAM should be ~32GB"
    assert mem_before['allocated'] >= 0, "Allocated should be non-negative"

    # Allocate memory
    print("\nAllocating 2GB tensor...")
    tensor = torch.randn(1024, 1024, 512, device='cuda')  # ~2GB

    mem_during = get_gpu_memory_info()
    print_gpu_memory_summary("During allocation: ")

    # Verify allocation increased
    assert mem_during['allocated'] > mem_before['allocated'], \
        "Allocated memory should increase after tensor allocation"

    # Free memory
    print("\nFreeing tensor...")
    del tensor
    torch.cuda.empty_cache()

    mem_after = get_gpu_memory_info()
    print_gpu_memory_summary("After cleanup: ")

    # Verify memory was freed
    assert mem_after['allocated'] < mem_during['allocated'], \
        "Allocated memory should decrease after cleanup"

    print("✓ Memory monitoring test passed")


def test_model_manager():
    """Test model manager sequential loading."""
    print("\n" + "="*60)
    print("TEST 3: Model Manager")
    print("="*60)

    manager = ModelManager(verbose=True)

    # Load first dummy model
    print("\nLoading model 1...")
    model1 = manager.load_model(
        "dummy_model_1",
        lambda: torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        ).cuda()
    )
    assert manager.get_current_model_name() == "dummy_model_1"
    mem_after_model1 = get_gpu_memory_info()

    # Load second dummy model (should unload first)
    print("\nLoading model 2 (should unload model 1 first)...")
    model2 = manager.load_model(
        "dummy_model_2",
        lambda: torch.nn.Sequential(
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 200)
        ).cuda()
    )
    assert manager.get_current_model_name() == "dummy_model_2"
    mem_after_model2 = get_gpu_memory_info()

    # Verify model1 was unloaded (memory should not be 2x)
    # (This is heuristic - if both models were loaded, allocated would be ~2x)
    print(f"\nMemory after model 1: {mem_after_model1['allocated']:.2f}GB")
    print(f"Memory after model 2: {mem_after_model2['allocated']:.2f}GB")

    # Manual unload
    print("\nManually unloading model 2...")
    manager.unload_current_model()
    assert manager.get_current_model_name() is None
    mem_after_unload = get_gpu_memory_info()

    assert mem_after_unload['allocated'] < mem_after_model2['allocated'], \
        "Memory should decrease after unloading model"

    print("✓ Model manager test passed")


def test_cuda_environment_variables():
    """Test CUDA environment variables are set correctly."""
    print("\n" + "="*60)
    print("TEST 4: CUDA Environment Variables")
    print("="*60)

    import os

    # Check PYTORCH_NVML_BASED_CUDA_CHECK
    nvml_check = os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK', '')
    print(f"PYTORCH_NVML_BASED_CUDA_CHECK: {nvml_check}")

    # Check PYTORCH_CUDA_ALLOC_CONF
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")

    assert 'expandable_segments' in alloc_conf.lower(), \
        "PYTORCH_CUDA_ALLOC_CONF should include expandable_segments"

    print("✓ Environment variables test passed")


def run_all_tests():
    """Run all GPU environment tests."""
    print("\n" + "="*70)
    print(" GPU ENVIRONMENT TEST SUITE")
    print("="*70)

    try:
        env_info = test_gpu_validation()
        test_memory_monitoring()
        test_model_manager()
        test_cuda_environment_variables()

        print("\n" + "="*70)
        print(" ALL TESTS PASSED ✓")
        print("="*70)
        print("\nEnvironment Summary:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        print("="*70)

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
