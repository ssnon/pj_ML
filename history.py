import torch
import matplotlib.pyplot as plt


def load_history(file_path):
    history = torch.load(file_path)
    train_loss_history = history['train_loss_history']
    # train_accuracy_history = history['train_accuracy_history']
    test_loss_history = history['test_loss_history']
    # test_accuracy_history = history['test_accuracy_history']
    return train_loss_history, test_loss_history

train_loss_history, test_loss_history = load_history('resnet18k_cifar10_sgd/noise-0.0_datasize-1.0/history.pth')

plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
