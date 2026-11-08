import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
ON_GPU = DEVICE.type == "cuda"

HIDDEN_ATOL = 2e-3 if ON_GPU else 1e-5
PROB_ATOL = 1e-4 if ON_GPU else 1e-5
GRAD_GLOBAL_RELATIVE_L2 = 1e-3 if ON_GPU else 1e-5
GRAD_ELEMENT_ATOL = 5e-3 if ON_GPU else 2e-4
GRAD_TENSOR_RELATIVE_L2 = 1e-2
GRAD_GLOBAL_FLOOR = 1e-4
