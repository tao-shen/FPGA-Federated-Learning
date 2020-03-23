# FPGA-Federated-Learning

## Knowedge Distillation baseline settings:

- Running on Cuda, Device: GTX2080Ti, Python 3.6.9

- Dataset: Cifar10

- Teacher Model: ResNet18 (acc=94.81%) LeNet5 (acc=61.91%)

- ```python
  epochs = trange(60)
  lr=1e-1, momentum=0.9, weight_decay=5e-4 #SGD
  StepLR(optimizer, step_size=20, gamma=0.1)
  ```

- Distillation settings: T=6, alpha=0.5

## Results

- Training time each epoch

  | Student\|Teacher | ResNet18  | LeNet5    | None      |
  | ---------------- | --------- | --------- | --------- |
  | ResNet18         | 50s/epoch | 26s/epoch | 25s/epoch |
  | LeNet5           | 25s/epoch | 4s/epoch  | 4s/epoch  |

- Total time and accuracy

  | Student\|Teacher | ResNet18       | LeNet5         | None           |
  | ---------------- | -------------- | -------------- | -------------- |
  | ResNet18         | 52:47 (94.12%) | 27:34 (78.24%) | 27:11 (89.27%) |
  | LeNet5           | 26:36 (62.63%) | 05:30 (59.95%) | 05:25 (57.83%) |
