import pytest
import subprocess
from importlib.metadata import distributions

def pytest_report_header(config):
    import datetime
    import platform
    import sys
    return [
        f"Test run: {datetime.datetime.now().isoformat()}",
        f"Hostname: {platform.node()}",
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {sys.version.split()[0]}"
    ]

def test_module_is_installed():
    packages = [dist.metadata.get("Name") for dist in distributions()]
    required = ["empanada-napari", "torch", "napari"]
    missing = [pkg for pkg in required if pkg not in packages]
    assert not missing, f"Missing packages: {', '.join(missing)}"

def test_module_imports():
    try:
        import napari
        import torch
        import empanada_napari
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")

@pytest.mark.dependency(name="nvidia_driver")
def test_nvidia_driver_available():
    try:
        result = subprocess.check_output(
            ["nvidia-smi"], 
            stderr=subprocess.STDOUT,
            timeout=2
        )
        assert "CUDA" in str(result)
    except Exception as e:
        pytest.fail(f"NVIDIA driver not found ({e}) - GPU acceleration unavailable")

@pytest.mark.dependency()
def test_torch_cuda_available():
    import torch
    if torch.version.cuda is None:
        pytest.skip("PyTorch not built with CUDA - GPU acceleration unavailable")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available to PyTorch - GPU acceleration unavailable")
    
    print(f"\nPyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")

def test_display_available():
    import os
    display = os.environ.get("DISPLAY")
    
    if not display:
        pytest.fail("DISPLAY unset - napari GUI unavailable")
    
    # xset, xrandr will error if can't access the display
    try:
        subprocess.run(
            ["xset", "q"],
            env={"DISPLAY": display},
            capture_output=True,
            timeout=2,
            check=True
        )
    except FileNotFoundError:
        pytest.fail(f"DISPLAY={display} but cannot verify (xset not found)")
    except subprocess.CalledProcessError:
        pytest.fail(f"Cannot connect to DISPLAY={display}")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timeout connecting to DISPLAY={display}")

