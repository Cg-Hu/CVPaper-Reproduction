import torch

loss = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [2.5, 3.6, 3.0]])
tensor1 = torch.tensor([2, 0, 1], dtype= torch.float32)
tensor2 = torch.tensor([2, 1, 1], dtype=torch.float32)
print(tensor1)
sum = torch.sum(tensor1 == tensor2)
print(sum.item())

