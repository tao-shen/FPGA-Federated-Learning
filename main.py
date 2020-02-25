import torch
import torchvision
from torchvision import transforms
from tqdm import trange
import Model
from Train import train
from Validate import validate
from Node import Node

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on', device)
model1 = Model.ResNet18()
model2 = Model.ResNet18()
model1.to(device)
model2.to(device)

tra_transformer = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
val_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="~/data", train=True, download=False, transform=tra_transformer)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_set = torchvision.datasets.CIFAR10(root="~/data", train=False, download=False, transform=val_transformer)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

node1 = Node(Model.ResNet18(), train_loader)
optimizer = torch.optim.SGD(node1.model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
epochs = trange(60)
loss = 0
acc = 0
acc_best = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in epochs:
    description = "Total Process (the {:d}-epoch): Loss = {:.4f} Accuracy = {:.2f}%".format(epoch + 1, loss, acc)
    epochs.set_description(description)
    train(node1, device, optimizer)
    loss, acc = validate(node1.model, device, test_loader)
    scheduler.step()
    msg = "Validation [{:d}-epoch]: val_Loss = {:.4f} val_Accuracy = {:.2f}%\n".format(epoch + 1, loss, acc)
    epochs.write(msg)
    if acc > acc_best:
        acc_best = acc
        epochs.write("A Better Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
        torch.save(node1.model.state_dict(), "Node1_ResNet18.pt")

print("Finished! The Best Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
