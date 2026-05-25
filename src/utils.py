import matplotlib.pyplot as plt
import random
import numpy as np


####################
# CONFUSION MATRIX
#####################

def show_confusionMatrix(cm, class_names=None):

    cm = cm.cpu().numpy()

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    # values inside cells
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(
                j, i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


#####################################
# LOSS + ACCURACY PLOT
####################################

def plot_metrics(trainLoss, testLoss, testAcc):

    epochs = range(1, len(trainLoss) + 1)

    plt.figure(figsize=(14, 5))

    # -----------------------
    # LOSS PLOT
    # -----------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainLoss, label="Train Loss")
    plt.plot(epochs, testLoss, label="Test Loss")

    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # -----------------------
    # ACCURACY PLOT
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, testAcc, label="Test Accuracy", color="green")

    plt.title("Test Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def show_random_predictions(model, loader, device, class_names, num_images=5):
    model.eval()

    images_list = []
    labels_list = []
    preds_list = []

    # collect one batch (or multiple batches if needed)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())

            if len(torch.cat(images_list)) >= num_images:
                break

    # flatten collected tensors
    images = torch.cat(images_list)[:num_images]
    labels = torch.cat(labels_list)[:num_images]
    preds = torch.cat(preds_list)[:num_images]

    # plot
    plt.figure(figsize=(12, 3))

    for i in range(num_images):
        img = images[i].squeeze()

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap="gray")

        plt.title(
            f"T:{class_names[labels[i]]}\nP:{class_names[preds[i]]}",
            fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
