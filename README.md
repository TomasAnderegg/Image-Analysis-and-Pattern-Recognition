# 🍫 Chocolate Recognition & Counting

### Deep Learning Challenge -- Image Analysis and Pattern Recognition (EE-451)

**EPFL -- May 2025**

This project implements a deep learning pipeline to **classify and count
multiple types of chocolates** in high-resolution images.

The task is formulated as a **multi-label regression problem**, where
the model predicts the count of each chocolate class present in an
image.

------------------------------------------------------------------------

# 📌 Project Overview

-   📚 Training set: 90 labeled images\
-   🧪 Test set: 180 unlabeled images\
-   🎯 Objective: Predict counts for 13 chocolate classes\
-   📏 Parameter constraint: ≤ 12M parameters\
-   🏆 Best Kaggle F1-score: **0.9534**\
-   🚀 Best internal lightweight model: **\~0.97**

------------------------------------------------------------------------

# 🧠 Model Architecture

## Backbone: ResNet18

We use a **ResNet18** CNN as feature extractor followed by a fully
connected layer for regression.

### Architecture Pipeline

Input Image (1024x1536)\
→ ResNet18 Convolutional Layers\
→ Global Average Pooling (512 features)\
→ Fully Connected Layer (13 outputs)\
→ Chocolate counts (regression output)

### Key Design Choices

-   Images resized from **6000×4000 → 1024×1536**
-   Output features: **512**
-   Total parameters: **11,183,181**
-   Skip connections prevent vanishing gradients
-   Global average pooling reduces spatial dimensions

A lightweight ResNet18 variant (\~2.8M parameters) was also tested and
achieved improved validation performance due to reduced overfitting.

------------------------------------------------------------------------

# 🔄 Training Pipeline

## 1️⃣ Data Preparation

-   Manual train/validation split:
    -   72 training images
    -   18 validation images
-   Uniform distribution of backgrounds, chocolate types, and foreign
    objects

------------------------------------------------------------------------

## 2️⃣ Data Augmentation

\~1152 augmented samples generated using:

-   RandomRotation (±7°)
-   ColorJitter (brightness, contrast, saturation)
-   Resize
-   Normalization (dataset mean & variance)

------------------------------------------------------------------------

## 3️⃣ Loss Function

**Mean Squared Error (MSE)**

Chosen because the task is regression-based (count prediction), not
single-label classification.

------------------------------------------------------------------------

## 4️⃣ Optimizer & Hyperparameters

-   Optimizer: **Adam**
-   Learning rate: 1e-3
-   Weight decay: 1e-4
-   Batch size: 4
-   Max epochs: 1000
-   Early stopping triggered at epoch **145**

------------------------------------------------------------------------

# 🔍 Inference Pipeline

1.  Load saved model (`best_model.pth`)
2.  Resize & normalize test images
3.  Forward pass through network
4.  Post-processing:
    -   Clip negative outputs to 0
    -   Round to nearest integer
    -   Convert to integer type
5.  Save predictions to `submission.csv`

Output format:

  Image_ID   Class_1   Class_2   ...   Class_13
  ---------- --------- --------- ----- ----------

------------------------------------------------------------------------

# 📊 Results & Milestones

## 🥉 Baseline (\~0.30 F1)

-   AlexNet
-   CrossEntropy Loss
-   No validation set
-   No augmentation

Used mainly to validate pipeline.

------------------------------------------------------------------------

## 🥈 Intermediate (\~0.68 F1)

-   ResNet18
-   MSE Loss
-   Adam + L2 regularization
-   \~6000 augmented samples

Significant improvement due to architecture upgrade and augmentation.

------------------------------------------------------------------------

## 🥇 Final Model (\~0.95 F1)

-   ResNet18
-   MSE Loss
-   Early stopping
-   Carefully selected validation split
-   Optimized augmentation
-   GPU acceleration (≈ ×50 faster training)

Best Kaggle score: **0.9534**

------------------------------------------------------------------------

# 📈 Qualitative Analysis

-   Validation F1 peaked at epoch 145
-   Early stopping prevented overfitting
-   Training loss decreased steadily
-   Slight validation divergence after optimal epoch

### Per-Class Performance

-   Perfect recall for all classes
-   Perfect precision except **Jelly Black**
-   Accuracy range: 94.4% -- 100%
-   Minor confusion between Jelly Black and Jelly Milk

------------------------------------------------------------------------

# 💡 Lessons Learned

### What Worked

-   ResNet18 backbone
-   MSE for regression
-   Careful validation strategy
-   Early stopping
-   Moderate augmentation

### Limitations

-   Suboptimal CutMix implementation
-   Early dataset imbalance
-   Sensitivity to foreign objects

------------------------------------------------------------------------

# 🔬 Future Improvements

-   Improved CutMix strategy
-   Progressive class-specific training
-   Better dataset balancing
-   Explore EfficientNet or attention mechanisms

------------------------------------------------------------------------

# ⚙️ Tech Stack

-   Python
-   PyTorch
-   NumPy
-   Pandas
-   Matplotlib
-   Kaggle

------------------------------------------------------------------------

# 👥 Authors

Group 10 -- EPFL

-   Rayan Bouchalouf\
-   Vincent Ellenrieder\
-   Tomas Garate Anderegg

------------------------------------------------------------------------

# 📜 Conclusion

A carefully tuned ResNet18 can achieve strong multi-label regression
performance under strict parameter constraints.\
Regularization and validation strategy proved more impactful than
increasing model complexity.

Final performance:

-   **Kaggle F1-score: 0.9534**
-   **Lightweight model: \~0.97**
