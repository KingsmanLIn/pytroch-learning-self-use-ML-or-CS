#张量的不同创建形式
import torch
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch1 = torch.tensor([1,2,3],device=device)
print(torch1)
#将numpy数字转换为tensor张量
np_array=np.array([1,3,5])
np_array[0]=100
tensor=torch.from_numpy(np_array).to(device)
print(tensor)
#修改np_array的值，会发现tensor的值也会发生变化，即tensor和np_array共享内存

#创造一个2D的张量
tensor_2d=torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
],device=device)
print(tensor_2d,tensor_2d.shape)
#创造3,4,5d的张量
# 创建 3D 张量（立方体）
tensor_3d = torch.stack([tensor_2d, tensor_2d + 10, tensor_2d - 5])  # 堆叠 3 个 2D 张量
print("3D Tensor (Cube):\n", tensor_3d)
print("Shape:", tensor_3d.shape)  # 形状

# 创建 4D 张量（向量的立方体）
tensor_4d = torch.stack([tensor_3d, tensor_3d + 100])  # 堆叠 2 个 3D 张量
print("4D Tensor (Vector of Cubes):\n", tensor_4d)
print("Shape:", tensor_4d.shape)  # 形状

# 创建 5D 张量（矩阵的立方体）
tensor_5d = torch.stack([tensor_4d, tensor_4d + 1000])  # 堆叠 2 个 4D 张量
print("5D Tensor (Matrix of Cubes):\n", tensor_5d)
print("Shape:", tensor_5d.shape)  # 形状

# 获取单元素值,只能从单元素张量进行提取
single_value = torch.tensor(42)
print("Single Element Value:", single_value.item())

#对张量进行数学操作（在CUDA中进行运算的时候应使用float进行运算）
#1、tensor_zl的取值
tensor_zl=torch.tensor([[1,2,3,4,5],[2,3,4,5,6]],dtype=torch.float32,device=device)
print(tensor_zl)
#取第一行的
print(tensor_zl[0])
#取第一行的第3个元素,第一个参数表述行，第二个参数表述行中列取值
print(tensor_zl[0][2])
#取第二列的所有元素
print(tensor_zl[:,1])
#其实与data.iloc[:,1]的效果是一样的,所以本质上就是数据切分的一种方式

#让张量形状进行变换,只能通过相同元素数量的张量矩阵进行变换
print(tensor_zl.view(2,5))
#将tensor_zl进行展开，展开成一行
print(tensor_zl.flatten())

#进行数学运算
#1、加法
tensor_add=tensor_zl+10
print(tensor_add)
#2、乘法
tensor_mul=tensor_zl*3
print(tensor_mul)
#3、内部元素求和
tensor_sum=tensor_zl.sum()
print(tensor_sum)
#4、张量之间的乘法（矩阵乘法）
tensor_matmul=torch.matmul(tensor_zl,tensor_mul.T)
print(tensor_matmul)
#张量当中元素的bool判断（之后应该是用来做遗忘和记忆，或者是对张量进行筛选，类似于激活函数），但是会展开成一个张量
tensor_bool=tensor_zl>3
print(tensor_bool)
tensor_filter=tensor_zl[tensor_zl>2]
print(tensor_filter)

#将张量转化为numpy,nupmpy只能在cpu上进行运算
tensor_chafnp=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
print(tensor_chafnp)
numpy_from_tensor=tensor_chafnp.numpy()
print(numpy_from_tensor)
tensor_chafnp[1][1]=100
print(tensor_chafnp)
print(numpy_from_tensor)
#若要不共享内存，则需要使用clone(),相当于复制一份现有结果放入numpy显示
tensor_chafnp=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
numpy_from_tensor=tensor_chafnp.clone().numpy()
tensor_chafnp[1][1]=100
print(tensor_chafnp)
print(numpy_from_tensor)

