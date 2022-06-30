import torch
import numpy as np

# list to tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# nparray to tensor : tensor and nparray are linked
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
np.add(np_array, 1, out=np_array)
print(f"x_np: {x_np}")
print(f"np_array: {np_array}")

# tensor to nparray : tensor and nparray are linked
t = torch.ones(5)
n = t.numpy()
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# get shape, dtype from defined tensor
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# rand, const tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# attr of tensor : shape / dtype / device
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# indexing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# concat
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# arith op
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# for single element tensor
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# update operand
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

