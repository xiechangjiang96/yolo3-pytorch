from nets.YOLOBody import YOLO3
import torch

model = YOLO3()
rand_data = torch.rand([1, 3, 416, 416])
small, medium, large = model(rand_data)
print(small.shape, medium.shape, large.shape)