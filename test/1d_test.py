import sys

sys.path.append("/home/osman/git_repos/ThermoBO")

# import numpy as np
# import torch
# from botorch.acquisition import qExpectedImprovement
# from botorch.optim import optimize_acqf_discrete

# from thermo_bo.data import DataParameters, get_data
# from thermo_bo.models import TorchGP, ExactGPModel
from thermo_bo.saasbo import SAASBO, SAASBOParameters

# df, properties = get_data(DataParameters("material_data", ["ultimate tensile strength (MPa)",
#                         "yield strength (MPa)"]))
# X = torch.tensor(
#                 df[list(properties)].to_numpy(), 
#                 dtype=torch.float64, device=torch.device("cpu"))
# y = torch.tensor(df[["yield strength (MPa)"]].to_numpy(), dtype=torch.float64,
#                  device=torch.device("cpu"))

# mc = MCGP(256, 128, 8)
# gp = mc.get_optimized_gp(X[:100], y[:100])
# EI = qExpectedImprovement(model=gp, best_f=y[:100].max())
# candidates, acq_values = optimize_acqf_discrete(EI, choices=X[100:], 
#             q=4, unique=True)
# print(candidates, acq_values)
# exact = ExactGPModel(1000)
# gp = exact.get_optimized_GP(X[:100], y[:100])
# print(gp)
saasbo_params = SAASBOParameters(gp="saas",
                                 acq_startegy="q",
                                 sampler="sobol",
                                 partitioning="non-dominated",
                                 seed_points=20,
                                 project_name="computational",
                                 targets=["Density (g/cm3)"],
                                 targets_mask=[False],
                                 active_features=None,
                                 desired_f=[0.33],
                                 device="cpu",
                                 n_iter=50,
                                 cand_size=1)
saasbo = SAASBO(saasbo_parameters=saasbo_params)
saasbo.run_optimization()
