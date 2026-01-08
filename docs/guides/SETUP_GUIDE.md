# Virtual Environment & TensorFlow Setup Guide

## Problem
TensorFlow is encountering a mutex lock error on the system:
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error: 
mutex lock failed: Invalid argument
```

This is typically caused by:
- Conflicting system-level TensorFlow installations
- Multiple Python processes accessing TensorFlow simultaneously
- Corrupted TensorFlow installation
- MacOS-specific threading issues

## Solution 1: Local Virtual Environment (Recommended for Local Machine)

### Automatic Setup

Run the automated setup script:

```bash
cd "/Users/rubayethassan/Desktop/424 project start"
./setup_venv.sh
```

This script will:
1. Create a fresh Python virtual environment (`vae_env`)
2. Install TensorFlow 2.15.0 (more stable version)
3. Install all required dependencies
4. Verify the installation

### Manual Setup

If you prefer manual setup:

```bash
# 1. Create virtual environment
python3 -m venv vae_env

# 2. Activate it
source vae_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install numpy==1.26.4 pandas==2.2.0 scikit-learn==1.4.0
pip install tensorflow==2.15.0
pip install matplotlib==3.8.2 seaborn==0.13.1
pip install shap==0.44.0

# 5. Verify installation
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

### Using the Virtual Environment

**Activate the environment:**
```bash
source vae_env/bin/activate
# Or use the helper script:
source activate_env.sh
```

**Run VAE training:**
```bash
python3 train_vae.py
```

**Deactivate when done:**
```bash
deactivate
```

### Quick Test

After activating the environment, test if TensorFlow works:

```bash
source vae_env/bin/activate
python3 test_vae_setup.py
```

If successful, run the quick training test:

```bash
python3 quick_train_vae.py
```

---

## Solution 2: Google Colab (Recommended for GPU Training)

Google Colab provides free GPU access and a clean Python environment. This is the **best option** for training deep learning models.

### Step-by-Step Colab Setup

#### 1. Upload Files to Google Drive

Upload these files to Google Drive (create a folder called `vae_project`):
- `config.py`
- `simple_tokenizer.py`
- `vae_model.py`
- `train_vae.py`
- `processed_data/` folder (all `.npy` and `.pkl` files)

#### 2. Create a Colab Notebook

Go to https://colab.research.google.com and create a new notebook.

#### 3. Enable GPU

- Click `Runtime` → `Change runtime type`
- Select `GPU` as Hardware accelerator
- Click `Save`

#### 4. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 5. Navigate to Project Directory

```python
import os
os.chdir('/content/drive/MyDrive/vae_project')
!ls -la
```

#### 6. Install Dependencies (if needed)

```python
# Most dependencies are pre-installed in Colab
# Only install if missing:
!pip install --upgrade tensorflow
```

#### 7. Run Training

```python
# Run the training script
!python train_vae.py
```

#### 8. Download Results

After training, download the results:

```python
from google.colab import files

# Download trained model
files.download('models/vae_best.keras')

# Download training plots
files.download('results/vae_training_history.png')
files.download('results/vae_latent_space.png')
```

### Alternative: Run Directly in Colab Cells

You can also copy the code directly into Colab cells:

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/vae_project')

# Cell 2: Import and configure
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Cell 3: Run training
!python train_vae.py
```

---

## Solution 3: Clean TensorFlow Reinstall (Local)

If virtual environment doesn't work, try completely removing and reinstalling TensorFlow:

```bash
# 1. Kill all Python processes
pkill -9 python3

# 2. Uninstall TensorFlow completely
pip3 uninstall tensorflow tensorflow-metal tensorflow-macos -y

# 3. Clear pip cache
pip3 cache purge

# 4. Restart terminal/close all terminal windows

# 5. Reinstall TensorFlow (in new terminal)
pip3 install tensorflow==2.15.0

# 6. Test
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Solution 4: Docker Container (Advanced)

For complete isolation:

```bash
# Pull TensorFlow Docker image
docker pull tensorflow/tensorflow:latest-gpu

# Run container with mounted volume
docker run -it --gpus all \
  -v "/Users/rubayethassan/Desktop/424 project start:/workspace" \
  tensorflow/tensorflow:latest-gpu bash

# Inside container:
cd /workspace
python3 train_vae.py
```

---

## Troubleshooting

### Issue: Virtual environment still has mutex error

**Solution**: The issue might be system-level. Try:
1. Restart your computer
2. Use Google Colab instead
3. Try different TensorFlow version: `pip install tensorflow==2.14.0`

### Issue: "No module named tensorflow"

**Solution**: Make sure virtual environment is activated:
```bash
source vae_env/bin/activate
which python3  # Should show path inside vae_env
```

### Issue: GPU not detected

**On Mac M1/M2**:
```bash
# Install Apple Silicon optimized version
pip install tensorflow-metal
```

**On Linux/Windows**:
```bash
pip install tensorflow[and-cuda]
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `config.py`:
```python
VAE_BATCH_SIZE = 32  # Instead of 64
```

Or use data subset:
```python
# In train_vae.py, modify data loading:
train_data = train_customer[:10000]  # Use only 10k samples
```

---

## Recommended Approach

🥇 **Best for GPU Training**: Use Google Colab (Solution 2)
- Free GPU access
- No setup issues
- Fast training
- Easy to share results

🥈 **Best for Local Development**: Virtual environment (Solution 1)
- Isolated dependencies
- Reproducible environment
- Works offline

🥉 **Quick Fix**: Clean reinstall (Solution 3)
- Fast to try
- Might resolve system conflicts

---

## Next Steps After Setup

Once TensorFlow is working:

1. **Test VAE Architecture**
   ```bash
   python3 test_vae_setup.py
   ```

2. **Quick Training Test (5 epochs)**
   ```bash
   python3 quick_train_vae.py
   ```

3. **Full Training (50 epochs)**
   ```bash
   python3 train_vae.py
   ```

4. **Monitor Training**
   ```bash
   # In another terminal:
   tensorboard --logdir logs/vae
   # Open http://localhost:6006
   ```

5. **Verify Results**
   - Check `models/vae_best.keras`
   - View `results/vae_training_history.png`
   - Review generated samples in terminal output

---

## Files in This Directory

- ✅ `setup_venv.sh` - Automated virtual environment setup script
- ✅ `activate_env.sh` - Quick activation helper (created after setup)
- ✅ `test_vae_setup.py` - Test TensorFlow without loading models
- ✅ `quick_train_vae.py` - Quick 5-epoch training test
- ✅ `train_vae.py` - Full training pipeline
- ✅ `vae_model.py` - VAE architecture
- ✅ All preprocessing files and data

Everything is ready to go once TensorFlow is working!
