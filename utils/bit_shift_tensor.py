from ap2 import approximate_power_of_two
import torch
import numpy as np


def extract_movement(bits: np.ndarray) -> np.ndarray:
    movement = np.where(bits < 0, -1 * np.log(1.0 + abs(bits)), np.log2(1.0 + bits))
    return movement


def bit_shift_tensor(tensor: torch.tensor) -> torch.tensor:
    if type(tensor) != torch.Tensor:
        raise TypeError("Bit shift function needs Tensor. Please transform")

    if torch.cuda.is_available():  # tensor to gpu
        tensor = tensor.type(torch.cuda.FloatTensor)
    tensor = approximate_power_of_two(tensor)  # calculate AP2
    tensor_np = tensor.cpu().numpy()
    tensor_np = tensor_np.astype(np.int64)  # reformat cuz bit shift

    flat_tensor = tensor_np.reshape(-1)
    flat_tensor = flat_tensor.astype(np.int64)
    # reshape for every various tensor. Some has 2 by 2 but
    # all tensor does not have 2 by 2

    flat_bits = extract_movement(flat_tensor)
    flat_bits = flat_bits.astype(np.int64)  # remake type

    flat_target = flat_tensor >> flat_bits  # bit shift
    flat_target = torch.from_numpy(flat_target)  # numpy to torch
    flat_target = flat_target.reshape(tensor_np.shape)  # reshape original shape

    return flat_target


if __name__ == "__main__":
    a = torch.randn(3, 3) * 10
    print(a)
    print("a.type : {}".format(type(a)))

    print(bit_shift_tensor(a))
    print("bit_shift_tensor.type : {}".format(type(bit_shift_tensor(a)[0])))
