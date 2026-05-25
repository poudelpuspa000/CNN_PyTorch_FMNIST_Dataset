from src.train import train_model
from src.evaluate import confusion_matrix, class_accuracy
from src.utils import plot_metrics, show_confusionMatrix

def main():

    (
        model,
        lossHistory,
        testLossHistory,
        accuracyHistory,
        device,
        testLoader
    ) = train_model(
        useAugmentation=False,
        batchSize=128,
        epochs=10
    )

     # LOSS + ACCURACY PLOTS
  

    plot_metrics(
        lossHistory,
        testLossHistory,
        accuracyHistory
    )

   
    # CONFUSION MATRIX
 

    cm = confusion_matrix(
        model,
        testLoader,
        device
    )

    print("\nConfusion Matrix:\n")
    print(cm.cpu().numpy())

    show_confusionMatrix(cm, class_names=classes)

    # CLASS ACCURACY
    

    classAccuracies = class_accuracy(cm)

    print("\nPer-Class Accuracy:\n")

    for className, accuracy in classAccuracies.items():
        print(f"{className}: {accuracy:.2f}%")
        
    show_random_predictions(
    model,
    testLoader,
    device,
    classes,
    num_images=5)


if __name__ == "__main__":
    main()
