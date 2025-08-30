import subprocess
import sys
import os

def check_autoflip_availability():
    """Check if AutoFlip is installed and accessible"""
    try:
        result = subprocess.run(['autoflip', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ AutoFlip is installed and accessible")
            return True
        else:
            print("✗ AutoFlip command failed")
            return False
    except FileNotFoundError:
        print("✗ AutoFlip not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ AutoFlip command timed out")
        return False

def install_autoflip():
    """Guide for AutoFlip installation"""
    print("\nAutoFlip Installation Guide:")
    print("1. Install Bazel: https://bazel.build/install")
    print("2. Clone MediaPipe: git clone https://github.com/google/mediapipe.git")
    print("3. Build AutoFlip:")
    print("   cd mediapipe")
    print("   bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/autoflip:run_autoflip")
    print("4. Copy binary to PATH or update AUTOFLIP_PATH environment variable")
    
def setup_fallback_mode():
    """Set up manual composition as AutoFlip alternative"""
    print("\nSetting up manual composition fallback...")
    
    # Update requirements for better manual composition
    fallback_requirements = """
# Enhanced requirements for manual composition
mediapipe>=0.10.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
protobuf>=3.20.0
scikit-image>=0.21.0  # For better image processing
scipy>=1.11.0         # For advanced filtering
"""
    
    with open('requirements_enhanced.txt', 'w') as f:
        f.write(fallback_requirements)
    
    print("✓ Enhanced requirements created: requirements_enhanced.txt")
    print("Run: pip install -r requirements_enhanced.txt")

def main():
    print("=== AutoFlip Setup Check ===")
    
    if check_autoflip_availability():
        print("\nAutoFlip is ready! You can use the full pipeline.")
    else:
        print("\nAutoFlip not available. Options:")
        print("A) Install AutoFlip (recommended for production)")
        print("B) Use enhanced manual composition (good for development)")
        
        choice = input("Choose (A/B): ").upper()
        
        if choice == 'A':
            install_autoflip()
        else:
            setup_fallback_mode()
            print("\nUsing manual composition mode for Day 3")

if __name__ == "__main__":
    main()