import torch

# 定义矩阵尺寸
dim1, dim2, dim3 = 50, 50, 50

# 生成随机矩阵
matrix1 = torch.randn(dim1, dim2)
matrix2 = torch.randn(dim2, dim3)

# 将矩阵移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix1 = matrix1.to(device)
matrix2 = matrix2.to(device)

# 定义每个block的尺寸
block_size = 16

# 计算每个维度上的block数量
num_blocks_dim1 = (dim1 + block_size - 1) // block_size
num_blocks_dim3 = (dim3 + block_size - 1) // block_size

# 创建结果张量并将其移动到GPU上
result = torch.zeros(dim1, dim3, device=device)

# 在GPU上执行矩阵乘法计算
for i in range(num_blocks_dim1):
    for j in range(num_blocks_dim3):
        # 计算当前block的起始和结束索引
        start_row = i * block_size
        end_row = min(start_row + block_size, dim1)
        start_col = j * block_size
        end_col = min(start_col + block_size, dim3)

        # 提取当前block对应的子矩阵
        sub_matrix1 = matrix1[start_row:end_row, :]
        sub_matrix2 = matrix2[:, start_col:end_col]

        # 执行矩阵乘法计算并将结果写入对应的位置
        result[start_row:end_row, start_col:end_col] = torch.matmul(sub_matrix1, sub_matrix2)

# 将结果移回CPU并打印
result = result.to("cp u")