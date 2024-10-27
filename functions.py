import torch
import numpy as np
from time import gmtime, strftime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import seaborn as sns
import random
import os
import torch.optim.lr_scheduler as lr_scheduler


class EarlyStopping:
    def __init__(self, patience=100, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        if os.path.isfile("./best_val_loss"):
            self.best_val_loss = torch.load("best_val_loss")
            print(f"Early stopping is active with patience: {patience}")
            print(f"Wont overwrite best model untill loss i lower than: {self.best_val_loss}")
        else:
            self.best_val_loss = 100

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            print("Resetet patience \n")
            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                torch.save(val_loss, "best_val_loss")
                if self.verbose:
                    print(f'Validation loss improved to {val_loss:.4f}. Saving model...')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered after {self.patience} epochs without improvement.')


def output_to_label(z):
    """Map network output z to a hard label {0, 1, 2, ...}

    Args:
        z (Tensor): Raw logits for each sample in a batch.
    Returns:
        c (Tensor): Predicted class label for each sample in a batch
    """
    return torch.argmax(z, dim=1)


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every, patience
):
    print("Starting training")
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    early_stopper = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, num_epochs + 1):
        model, train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn, train_loader, val_loader, device, print_every
        )
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
            f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
            f"Val. loss: {val_loss:.3f}, "
            f"Val. acc.: {val_acc:.3f},"
            f"Lr.: {scheduler.get_last_lr()}"
        )
        train_losses.append(sum(train_loss)/len(train_loss))
        train_accs.append(sum(train_acc)/len(train_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step()
        # Check for early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break  # Exit
    return model, train_losses, train_accs, val_losses, val_accs, epoch


def train_epoch(
    model, optimizer, loss_fn, train_loader, val_loader, device, print_every
):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model.forward(inputs)
        loss = loss_fn(z, labels)
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(z)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

        # If you want to print your progress more often than every epoch you can
        # set `print_every` to the number of batches you want between every status update.
        # Note that the print out will trigger a full validation on the full val. set => slows down training
        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            print(
                f"\tBatch {batch_index}/{num_batches}: "
                f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                f"\tTrain acc.: {sum(train_acc_batches[-print_every:])/print_every:.3f}, "
                f"\tVal. loss: {val_loss:.3f}, "
                f"\tVal. acc.: {val_acc:.3f},"
                f"\tTime:{strftime('%Y-%m-%d %H:%M:%S', gmtime())}"
            )

    return model, train_loss_batches, train_acc_batches


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)

            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            hard_preds = output_to_label(z)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum / len(val_loader), val_acc_cum / len(val_loader)


def train_val_plots(train_losses, train_accs, val_losses, val_accs, num_epochs):

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(1, num_epochs, len(train_losses)), train_losses, label='Training Loss', color='blue')
    plt.plot(np.linspace(1, num_epochs, len(val_losses)), val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(1, num_epochs, len(train_accs)), train_accs, label='Training Accuracy', color='blue')
    plt.plot(np.linspace(1, num_epochs, len(val_accs)), val_accs, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()


def confmatrix(model, test_dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(torch.device("cuda")), labels.to(torch.device("cuda"))
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    print(f"f1-score= {f1}, Recall: {recall}, Precision: {precision}")

    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
    # Calculate and plot confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')


def image_predictions(model, dataset, numberofimages):
    # Check if CUDA (GPU) is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        fig, axes = plt.subplots(1, numberofimages)
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
        for i in range(numberofimages):
            random_index = random.randint(0, len(dataset) - 1)
            image, label = dataset[random_index]

            image = image.to(device)
            image_input = image.unsqueeze(0)

            output = model(image_input)
            prediction = output.argmax(dim=1).item()

            # Move image back to CPU for display
            image_np = image.cpu().numpy().transpose(1, 2, 0)

            # Display the image in the ith subplot
            axes[i].imshow(np.squeeze(image_np), cmap="gray")
            axes[i].set_title(f"True: {labels[label]}, Predicted: {labels[prediction]}")
            axes[i].axis('off')

        plt.tight_layout()
