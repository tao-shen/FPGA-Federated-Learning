import torch
import torchvision
from torchvision import transforms
import Model
from Validate import validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on', device)
teacher = Model.ResNet18()
teacher.load_state_dict(torch.load('Teacher.pt'))
val_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.CIFAR10(root="~/data", train=False, download=False, transform=val_transformer)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

loss, acc = validate(teacher, device, test_loader)
print(acc)
