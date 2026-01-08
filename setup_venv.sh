#!/bin/bash
# Updated Virtual Environment Setup Script
# Compatible with Python 3.13 and latest TensorFlow

set -e  # Exit on error

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_NAME="vae_env"

echo "=========================================="
echo "Virtual Environment Setup (Python 3.13)"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $VENV_NAME"
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "$PYTHON_VERSION"

# Step 2: Remove old virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    echo ""
    echo "Step 2: Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Step 3: Create new virtual environment
echo ""
echo "Step 3: Creating new virtual environment..."
python3 -m venv "$VENV_NAME"

# Step 4: Activate virtual environment
echo ""
echo "Step 4: Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Step 5: Upgrade pip
echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip

# Step 6: Install dependencies
echo ""
echo "Step 6: Installing dependencies..."
echo "This may take a few minutes..."

# Install core dependencies
echo "Installing NumPy, Pandas, Scikit-learn..."
pip install numpy pandas scikit-learn

# Install TensorFlow (latest version compatible with Python 3.13)
echo ""
echo "Installing TensorFlow (latest version)..."
pip install tensorflow

# Install visualization libraries
echo ""
echo "Installing Matplotlib, Seaborn..."
pip install matplotlib seaborn

# Install SHAP for explainability (will be used later)
echo ""
echo "Installing SHAP..."
pip install shap

# Step 7: Verify installation
echo ""
echo "=========================================="
echo "Step 7: Verifying installation..."
echo "=========================================="

python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")
print("")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy error: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas error: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"✗ Scikit-learn error: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPU available: {len(gpus) > 0} ({len(gpus)} device(s))")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib error: {e}")

try:
    import seaborn
    print(f"✓ Seaborn version: {seaborn.__version__}")
except Exception as e:
    print(f"✗ Seaborn error: {e}")

print("")
print("✅ Dependency installation complete!")
EOF

# Step 8: Create activation helper script
echo ""
echo "Step 8: Creating activation helper script..."
cat > activate_env.sh << 'ACTIVATE_SCRIPT'
#!/bin/bash
# Quick activation script
source vae_env/bin/activate
echo "✅ Virtual environment activated!"
echo ""
echo "Python: $(python3 --version)"
echo "TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "To deactivate, run: deactivate"
ACTIVATE_SCRIPT

chmod +x activate_env.sh

# Step 9: Test TensorFlow in environment
echo ""
echo "=========================================="
echo "Step 9: Testing TensorFlow..."
echo "=========================================="

python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    
    # Try to create a simple model to test TensorFlow
    print("Creating a simple test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    print("✅ TensorFlow model creation successful!")
    
    # Test on dummy data
    import numpy as np
    x = np.random.random((10, 5))
    y = model.predict(x, verbose=0)
    print("✅ TensorFlow prediction successful!")
    print(f"   Output shape: {y.shape}")
    
except Exception as e:
    print(f"⚠️  TensorFlow test failed: {e}")
    print("   This may be a system-level issue.")
    print("   Consider using Google Colab for training.")
EOF

# Final instructions
echo ""
echo "=========================================="
echo "✅ Virtual Environment Setup Complete!"
echo "=========================================="
echo ""
echo "📌 To activate the environment:"
echo "   source vae_env/bin/activate"
echo ""
echo "📌 Or use the helper script:"
echo "   source activate_env.sh"
echo ""
echo "📌 To test VAE setup:"
echo "   source vae_env/bin/activate"
echo "   python3 test_vae_setup.py"
echo ""
echo "📌 To run quick training test:"
echo "   source vae_env/bin/activate"
echo "   python3 quick_train_vae.py"
echo ""
echo "📌 To run full training:"
echo "   source vae_env/bin/activate"
echo "   python3 train_vae.py"
echo ""
echo "📌 To deactivate:"
echo "   deactivate"
echo ""
echo "⚠️  If TensorFlow still has issues, try:"
echo "   - Google Colab (see VAE_Training_Colab.ipynb)"
echo "   - Different machine"
echo "   - Docker container"
echo ""
