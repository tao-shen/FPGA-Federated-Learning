import torch


def validate(model, device, test_loader):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss = total_loss + torch.nn.CrossEntropyLoss()(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss = total_loss / (idx + 1)
        acc = correct / len(test_loader.dataset) * 100
    return total_loss, acc
