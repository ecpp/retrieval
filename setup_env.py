#!/usr/bin/env python
"""
Environment setup script for the CAD part retrieval system.
This script checks for required dependencies and helps install them.
"""
import subprocess
import sys
import importlib.util
import pkg_resources

def check_package(package_name):
    """Check if a package is installed"""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_package(package_name, version=None):
    """Install a package using pip"""
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name
    
    print(f"Installing {package_spec}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])

def main():
    """Main function to check and install dependencies"""
    print("Checking dependencies for CAD part retrieval system...")
    
    # Required packages with minimum versions
    required_packages = {
        "torch": "2.0.0",
        "torchvision": "0.15.0",
        "transformers": "4.30.0",
        "faiss-cpu": "1.7.4",
        "pillow": "9.5.0",
        "numpy": "1.24.0",
        "tqdm": "4.65.0",
        "matplotlib": "3.7.0",
        "scikit-learn": "1.2.0",
        "pandas": "2.0.0",
        "pyyaml": "6.0"
    }
    
    # Check and install packages
    for package, min_version in required_packages.items():
        if not check_package(package):
            print(f"{package} is not installed.")
            install_package(package, min_version)
        else:
            installed_version = pkg_resources.get_distribution(package).version
            print(f"{package} is installed (version {installed_version}).")
    
    # Check for CUDA support in PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
        
        print(f"\nPyTorch CUDA support:")
        print(f"CUDA available: {cuda_available}")
        print(f"Device count: {device_count}")
        print(f"Device name: {device_name}")
    except Exception as e:
        print(f"Error checking CUDA support: {e}")
    
    print("\nSetup complete! You should now be able to run the CAD part retrieval system.")
    print("Try running 'python demo.py' to test the system.")

if __name__ == "__main__":
    main()
