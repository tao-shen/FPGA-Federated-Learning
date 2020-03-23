import torch.nn.functional as F
from tqdm import tqdm


def train(node, device, optimizer):
    model1 = node.model.to(device)
    train_loader = node.local_data
    model1.train()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"

    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(device), target.to(device)
            pred = model1(data)
            loss1 = F.cross_entropy(pred, target)
            loss1.backward()
            optimizer.step()
            total_loss += loss1
            avg_loss = total_loss / (idx + 1)
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
