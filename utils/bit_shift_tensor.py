from ap2 import approximate_power_of_two
import torch
import numpy as np


def bit_shift_tensor(tensor: torch.tensor):
    if type(tensor) != torch.Tensor:
        raise TypeError("Bit shift function needs Tensor. Please transform")

    if torch.cuda.is_available():               #tensor to gpu
        tensor = tensor.type(torch.cuda.FloatTensor)
    tensor = approximate_power_of_two(tensor)  # calculate AP2

    if torch.cuda.is_available():           #there is two type
        tensor_np = tensor.cpu().numpy()
    else:
        tensor_np = tensor.numpy()          # reformat cuz bit shift

    tensor_np = tensor_np.astype(np.int64)  # reformat cuz bit shift

    flat_tensor = tensor_np.reshape(-1)  # reshape for every various tensor. Some has 2 by 2 but
    # all tensor does not have 2 by 2

    flat_bits = np.zeros(flat_tensor.shape)  # make numpy array
    flat_bits = flat_bits.astype(np.int64)  # remake type

    for arr_len in range(len(flat_tensor)):  # calculation log2
        if flat_tensor[arr_len] < 0:
            flat_bits[arr_len] = -1 * np.log2(1 + abs(flat_tensor[arr_len]))
        else:
            flat_bits[arr_len] = np.log2(1 + flat_tensor[arr_len])

    flat_target = flat_tensor >> flat_bits  # bit shift
    flat_target = torch.from_numpy(flat_bits)  # numpy to torch
    flat_target = flat_target.reshape(tensor_np.shape)  # reshape original shape

    return flat_target


if __name__ == "__main__":
    a = torch.randn(3,3)
    print(a)
    print("a.type : {}".format(type(a)))
    print(bit_shift_tensor(a))
    print("bit_shift_tensor.type : {}".format(type(bit_shift_tensor(a)[0])))
