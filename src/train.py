from src.data import get_dataloaders
from src.model import Model
from src.evaluate import evaluate_accuracy, evaluate_loss

import torch
import torch.optim as optim
import torch.nn as nn
import os
# Create models directory if it does not exist
#os.makedirs("models", exist_ok=True)
def train_model(
    useAugmentation=False,
    batchSize=128,
    epochs=10
):

    print("Starting training process...\n")

    criterion = nn.CrossEntropyLoss()

    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"Using device: {device}\n")

    net = Model().to(device)

    trainLoader, testLoader = get_dataloaders(
        useAugmentation,
        batchSize
    )

    optimizer = optim.Adam(
        net.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )

    lossHistory = []
    testLossHistory = []
    accuracyHistory = []

    bestAccuracy = 0.0
    bestModelState = None

    ####################
    # TRAINING LOOP
    #####################

    for epoch in range(epochs):

        net.train()

        runningLoss = 0.0

        for inputs, labels in trainLoader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            runningLoss += loss.item()

        scheduler.step()

        # Average training loss
        trainLoss = runningLoss / len(trainLoader)

        # Evaluate test loss
        testLoss = evaluate_loss(
            net,
            testLoader,
            criterion,
            device
        )

        # Evaluate accuracy
        testAccuracy = evaluate_accuracy(
            net,
            testLoader,
            device
        )

        # Save best model
        if testAccuracy > bestAccuracy:
            bestAccuracy = testAccuracy
            bestModelState = net.state_dict()
            #savePath = os.path.join("models", "best_fashionmnist_model.pth")
            #torch.save(net.state_dict(), savePath)
            #print(f"Best model saved at: {savePath}")

        # Save histories
        lossHistory.append(trainLoss)
        testLossHistory.append(testLoss)
        accuracyHistory.append(testAccuracy)

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {trainLoss:.4f} | "
            f"Test Loss: {testLoss:.4f} | "
            f"Test Accuracy: {testAccuracy:.2f}%"
        )

    # Load best model
    net.load_state_dict(bestModelState)

    print("\nTraining finished.")
    print(f"Best Test Accuracy: {bestAccuracy:.2f}%")

    return (
        net,
        lossHistory,
        testLossHistory,
        accuracyHistory,
        device,
        testLoader
    )
