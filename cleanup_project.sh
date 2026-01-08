#!/bin/bash
# Project Cleanup and Reorganization Script
# This script will organize the project structure for GitHub and report writing

echo "=========================================="
echo "PROJECT CLEANUP AND REORGANIZATION"
echo "=========================================="
echo ""

# Create organized directory structure
echo "Creating organized directory structure..."
mkdir -p src/models
mkdir -p src/training
mkdir -p src/evaluation
mkdir -p src/utils
mkdir -p src/inference
mkdir -p docs/guides
mkdir -p docs/summaries
mkdir -p docs/comparisons
mkdir -p experiments/demos
mkdir -p experiments/tests

echo "✓ Directory structure created"
echo ""

# ============================================
# CORE SOURCE CODE (src/)
# ============================================
echo "Organizing core source code..."

# Models
mv config.py src/
mv vae_model.py src/models/
mv gan_model.py src/models/
mv hybrid_model.py src/models/
mv simple_tokenizer.py src/utils/

# Training
mv train_vae.py src/training/01_train_vae.py
mv train_hybrid.py src/training/02_train_hybrid.py
mv fine_tune_with_diversity.py src/training/03_fine_tune_with_diversity.py
mv train_with_fairness.py src/training/04_train_with_fairness.py

# Evaluation
mv evaluation_metrics.py src/evaluation/
mv diversity_metrics.py src/evaluation/
mv bias_mitigation.py src/evaluation/
mv explainability.py src/evaluation/

# Utilities
mv preprocessing.py src/utils/
mv visualizations.py src/utils/
mv compare_input_approaches.py src/evaluation/model_comparison.py

# Inference
mv inference_with_diversity.py src/inference/

echo "✓ Core source code organized"
echo ""

# ============================================
# DOCUMENTATION (docs/)
# ============================================
echo "Organizing documentation..."

# Main guides
mv COMPREHENSIVE_MODEL_COMPARISON_REPORT.md docs/MODEL_COMPARISON_REPORT.md
mv FINE_TUNING_GUIDE.md docs/guides/
mv EVALUATION_GUIDE.md docs/guides/
mv SETUP_GUIDE.md docs/guides/
mv XAI_README.md docs/guides/EXPLAINABILITY_GUIDE.md
mv BIAS_README.md docs/guides/BIAS_MITIGATION_GUIDE.md

# Comparison reports
mv INPUT_APPROACH_COMPARISON_RESULTS.md docs/comparisons/
mv COMPARISON_REPORT_DELIVERABLES.md docs/comparisons/

# Summaries
mv DIVERSITY_METRICS_SUMMARY.md docs/summaries/
mv DIVERSITY_TRAINING_SUMMARY.md docs/summaries/
mv BIAS_MITIGATION_SUMMARY.md docs/summaries/
mv XAI_IMPLEMENTATION_SUMMARY.md docs/summaries/
mv VAE_ARCHITECTURE_SUMMARY.md docs/summaries/
mv GAN_ARCHITECTURE_SUMMARY.md docs/summaries/

echo "✓ Documentation organized"
echo ""

# ============================================
# EXPERIMENTS (experiments/)
# ============================================
echo "Organizing experiments and demos..."

# Demos
mv demo_vae_training.py experiments/demos/
mv demo_hybrid_training.py experiments/demos/
mv demo_bias_mitigation.py experiments/demos/
mv demo_explainability.py experiments/demos/
mv demo_diversity_training.py experiments/demos/

# Tests
mv test_vae_setup.py experiments/tests/
mv test_vae_architecture.py experiments/tests/
mv verify_preprocessing.py experiments/tests/
mv check_preprocessing.py experiments/tests/

# Proof of concept
mv proof_of_concept.py experiments/
mv dataset_visualizations.py experiments/

echo "✓ Experiments organized"
echo ""

# ============================================
# DELETE UNNECESSARY FILES
# ============================================
echo "Removing unnecessary/duplicate files..."

# Remove duplicate summaries
rm -f TRAINING_PIPELINE_SUMMARY.md
rm -f VAE_TRAINING_SUMMARY.md
rm -f VAE_IMPLEMENTATION_GUIDE.md
rm -f PREPROCESSING_SUMMARY.md
rm -f EVALUATION_SUMMARY.md
rm -f VISUALIZATION_SUMMARY.md
rm -f SHAP_EXPLAINABILITY_SUMMARY.md
rm -f ARCHITECTURE_DIAGRAMS.md
rm -f SUCCESS_SUMMARY.md
rm -f COMPLETE_PROJECT_SUMMARY.md
rm -f PROJECT_COMPLETION_SUMMARY.md
rm -f FINAL_DELIVERY_CHECKLIST.md
rm -f CODE_CLEANUP_SUMMARY.md
rm -f MODEL_COMPARISON_SUMMARY.md
rm -f TEXT_EXAMPLES_ANALYSIS.md
rm -f TASK2_POC_VISUALIZATIONS_SUMMARY.md
rm -f BIAS_MITIGATION_GUIDE.md

# Remove duplicate/test scripts
rm -f quick_train_vae.py
rm -f quick_comparison_demo.py
rm -f model_comparison_demo.py
rm -f model_comparison.py
rm -f generate_comparison_report.py
rm -f advanced_visualizations.py
rm -f show_text_examples.py
rm -f shap_word_importance_demo.py
rm -f demo_diversity_tracking.py
rm -f fine_tune_hybrid.py
rm -f train_vae_complete.py
rm -f DIVERSITY_QUICK_REFERENCE.py
rm -f VISUALIZATION_QUICK_REFERENCE.md

# Remove shell scripts (keep only setup)
rm -f activate_env.sh
rm -f monitor_training.sh

# Remove Jupyter notebook (converted to Python scripts)
rm -f VAE_Training_Colab.ipynb

# Remove macOS files
rm -f .DS_Store

# Clean pycache
rm -rf __pycache__

echo "✓ Unnecessary files removed"
echo ""

# ============================================
# CREATE README AND PROJECT STRUCTURE
# ============================================
echo "Creating project documentation..."

echo "✓ Cleanup and reorganization complete!"
echo ""
echo "New project structure:"
echo "  src/"
echo "    ├── models/           # Model architectures (VAE, GAN, Hybrid)"
echo "    ├── training/         # Training scripts"
echo "    ├── evaluation/       # Evaluation and metrics"
echo "    ├── utils/            # Utilities (preprocessing, tokenizer)"
echo "    └── inference/        # Production inference"
echo "  docs/"
echo "    ├── guides/           # User guides"
echo "    ├── summaries/        # Technical summaries"
echo "    └── comparisons/      # Model comparison reports"
echo "  experiments/"
echo "    ├── demos/            # Demo scripts"
echo "    └── tests/            # Test scripts"
echo "  data/"
echo "    └── synthetic_it_support_tickets.csv"
echo "  results/              # Generated results"
echo "  models/               # Saved model weights"
echo "  logs/                 # Training logs"
echo ""
