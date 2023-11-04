import torch

def matrix_multiply_cpu(matrix1, matrix2, dim1, dim2, dim3):
    result = torch.zeros(dim1, dim3)
    for i in range(dim1):
        for j in range(dim3):
            for k in range(dim2):
                result[i, j] += matrix1[i, k] * matrix2[k, j]
    return result

# 定义矩阵尺寸
dim1, dim2, dim3 = 5, 5, 5

# 生成随机矩阵
matrix1 = torch.randn(dim1, dim2)
matrix2 = torch.randn(dim2, dim3)

# 调用CPU版本矩阵乘法
result_cpu = matrix_multiply_cpu(matrix1, matrix2, dim1, dim2, dim3)
print("\n")
print("Result  (CPU):\n", result_cpu)