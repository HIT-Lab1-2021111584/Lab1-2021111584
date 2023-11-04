import torch

def matrix_multiply_gpu(matrix1, matrix2, dim1, dim2, dim3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix1 = matrix1.to(device)
    matrix2 = matrix2.to(device)
    result = torch.zeros(dim1, dim3, device=device)
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

# 调用GPU版本矩阵乘法
result_gpu = matrix_multiply_gpu(matrix1, matrix2, dim1, dim2, dim3)
print("hello world")
print("Result (GPU):\n", result_gpu)
print("\n")