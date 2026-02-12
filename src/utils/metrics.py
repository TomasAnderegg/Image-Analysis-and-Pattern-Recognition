import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def imagewise_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the image-wise average F1-score.

    This function calculates the F1-score for each image (row) in the input arrays,
    then returns the mean F1-score across all images. The F1-score is computed as:
        F1 = 2 * TP / (2 * TP + FP + FN)
    where TP is the number of true positives, FP is false positives, and FN is false negatives.
    If both the ground truth and prediction are all zeros for an image, the F1-score is set to 1.0.

    Args:
        y_true (np.ndarray): Ground truth binary array of shape (N, C), where N is the number of images and C is the number of classes.
        y_pred (np.ndarray): Predicted binary array of shape (N, C).

    Returns:
        float: The average F1-score across all images.
    """
    N, C = y_true.shape
    f1s = []

    for i in range(N):
        # True Positives: count of correct positive predictions for image i
        TP = np.sum(np.minimum(y_true[i], y_pred[i]))
        # False Positives + False Negatives: count of mismatches for image i
        FPN = np.sum(np.abs(y_true[i] - y_pred[i]))

        # If both prediction and ground truth are all zeros, consider it a perfect match
        if 2 * TP + FPN == 0:
            f1 = 1.0
        else:
            f1 = (2 * TP) / (2 * TP + FPN)
        f1s.append(f1)

    # Return the mean F1-score across all images
    return np.mean(f1s)

def plot_metrics(train_losses, val_losses, train_f1s, val_f1s, save_path="metrics_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss over Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1s, label="Train F1")
    plt.plot(epochs, val_f1s, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Metrics curves saved to {save_path}")

def plot_multilabel_confusion_matrices(y_true, y_pred, class_names, save_dir="conf_matrices"):
    os.makedirs(save_dir, exist_ok=True)
    for i, class_name in enumerate(class_names):
        y_true_bin = (y_true[:, i] > 0).astype(int)
        y_pred_bin = (y_pred[:, i] > 0).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {class_name}")
        plt.colorbar()
        plt.xticks([0, 1], ["Not Present", "Present"])
        plt.yticks([0, 1], ["Not Present", "Present"])
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add numbers in each cell
        thresh = cm.max() / 2.
        for x in range(cm.shape[0]):
            for y in range(cm.shape[1]):
                plt.text(
                    y, x, format(cm[x, y], 'd'),
                    ha="center", va="center",
                    color="white" if cm[x, y] > thresh else "black"
                )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/conf_matrix_{class_name}.png")
        plt.close()
    print(f"Saved {len(class_names)} confusion matrices to {save_dir}")