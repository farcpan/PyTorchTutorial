import numpy as np
import time
import torch

"""
https://yutaroogawa.github.io/pytorch_tutorials_jp/
PyTorch入門 1. テンソル
https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/0_Learn%20the%20Basics/0_1_tensors_tutorial_js.ipynb#scrollTo=A6_nStSgo65o
"""

def main():
    # array -> tensor
    data = [[1, 2], [3, 4]]
    tensor_data = torch.tensor(data)
    print(f"data -> tensor {tensor_data}")

    # numpy array -> tensor
    np_array = np.array(data)
    print(f"numpy_array {np_array}")
    tensor_data_from_np_array = torch.from_numpy(np_array)
    print(f"numpy_array -> tensor {tensor_data_from_np_array}")

    # one initialized tensor
    ones_tensor = torch.ones_like(tensor_data)  # tensor_dataと同じ構造を持つ全要素が1で初期化されたTensor
    print(f"ones {ones_tensor}")

    # random initialized tensor
    random_tensor = torch.rand_like(tensor_data, dtype=torch.float)
    print(f"random {random_tensor}")

    # shaped tensor
    shape = (3, 2, 1, )
    random_tensor_from_shape = torch.rand(shape)
    print(f"random tensor from shape {random_tensor_from_shape}")

    # tensor attirbutes
    print(f"tensor shape: {random_tensor_from_shape.shape}")
    print(f"tensor data type: {random_tensor_from_shape.dtype}")
    print(f"which device tensor on: {random_tensor_from_shape.device}")

    # GPU device
    if torch.cuda.is_available():
        tensor_on_gpu = random_tensor_from_shape.to('cuda')
        print(f"which device tensor on: {tensor_on_gpu.device}")
    else:
        print("GPU cannot be used on this computer.")
    
    # tensor operation
    tensor_v2 = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]])
    print(f"shape: {tensor_v2.shape}")
    print(f"1st row: {tensor_v2[0]}")
    print(f"last row: {tensor_v2[-1]}")
    print(f"1st column: {tensor_v2[:, :, 0]}")
    print(f"last column: {tensor_v2[:, :, -1]}")

    # NxN matrix
    N = 100
    nxn_matrix_shape = (N, N)
    matrix_A = torch.rand(nxn_matrix_shape)
    matrix_B = torch.rand(nxn_matrix_shape)
    matrix_A_gpu = matrix_A.to("cuda")
    matrix_B_gpu = matrix_B.to("cuda")

    start = time.time()
    matrix_AB_v1 = matrix_A @ matrix_B.T
    matrix_AB_v2 = matrix_A.matmul(matrix_B)
    #print(f"AB v1: {matrix_AB_v1}")
    #print(f"AB v2: {matrix_AB_v2}")
    elapsed_time = time.time() - start
    print(f"Elapsed time: {elapsed_time} [sec]")

    start = time.time()
    matrix_AB_v1_gpu = matrix_A_gpu @ matrix_B_gpu.T
    matrix_AB_v2_gpu = matrix_A_gpu.matmul(matrix_B_gpu)
    #print(f"AB v1: {matrix_AB_v1_gpu}")
    #print(f"AB v2: {matrix_AB_v2_gpu}")
    elapsed_time = time.time() - start
    print(f"Elapsed time (GPU): {elapsed_time} [sec]")

    # summation of tensor
    agg = matrix_AB_v1_gpu.sum()
    agg_item = agg.item()
    print(f"agg_item: {agg_item}")

    # tensor -> numpy
    numpy_array_from_tensor = matrix_AB_v1_gpu.cpu().numpy()
    print(f"numpy array from tensor: {numpy_array_from_tensor}")
    

if __name__ == '__main__':
    main()
