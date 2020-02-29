# FPGA-Federated-Learning

## Knowedge Distillation baseline settings:

- Running on Cuda, Device: GTX2080Ti, Python 3.6.9

- Dataset: Cifar10

- Teacher Model: ResNet18 (acc=94.81%)

- ```python
  epochs = trange(60)
  lr=1e-1, momentum=0.9, weight_decay=5e-4 #SGD
  StepLR(optimizer, step_size=20, gamma=0.1)
  ```

- Distillation settings: T=6, alpha=0.5

## Results

- Student Model: ResNet18 (result_acc=93.71%)
- Training time: each_epoch: 48s, total_time: 49:58

- Normal Training: ResNet18 (result_acc=93.05%)
- Training time: each_epoch: 25s, total_time: 26:34




