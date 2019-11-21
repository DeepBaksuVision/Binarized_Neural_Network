import torch


def approximate_power_of_two(x: torch.tensor):
    with torch.no_grad():
        return torch.sign(x) * 2 ** torch.round(torch.log2(torch.abs(x)))


if __name__ == "__main__":
    print(approximate_power_of_two(torch.randn(3,3)))
