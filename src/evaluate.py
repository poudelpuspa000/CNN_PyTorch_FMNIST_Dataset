from src.data import classes
################
# ACCURACY
#################

def evaluate_accuracy(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


#############
# LOSS
#############

def evaluate_loss(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


#######################
# CONFUSION MATRIX
#######################

def confusion_matrix(model, loader, device):

    num_classes = len(classes)

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels, preds):
                cm[t, p] += 1

    return cm


###################
# CLASS ACCURACY
###################

def class_accuracy(confusionMatrix):

    classAccuracies = {}

    for i, className in enumerate(classes):

        correct = confusionMatrix[i][i].item()
        total = confusionMatrix[i].sum().item()

        classAccuracies[className] = (
            100 * correct / total if total > 0 else 0
        )

    return classAccuracies
