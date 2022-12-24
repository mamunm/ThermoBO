import sys
sys.path.append("/home/osman/git_repos/ThermoBO")

import torch
import numpy as np
from thermo_bo.data import get_data, DataParameters
from thermo_bo.models import MCGP, ExactGP
from thermo_bo.saasbo import SAASBOParameters, SAASBO

df, properties = get_data(DataParameters("material_data", ["yield strength max (MPa)", 
                        "ultimate tensile strength min (MPa)"]))
print(df)
# X = torch.tensor(
#                 df[[i for i in df.columns if i != "yield strength (MPa)"]].to_numpy(), 
#                 dtype=torch.float64, device=torch.device("cpu"))
# y = torch.tensor(df[["yield strength (MPa)"]].to_numpy(), dtype=torch.float64,
#                  device=torch.device("cpu"))

# mc = MCGP(256, 128, 8)
# gp = mc.get_optimized_gp(X, y)
# print(gp)
# exact = ExactGP(1000)
# gp = exact.get_optimized_GP(X[:100], y[:100])
# print(gp)

# saasbo_params = SAASBOParameters(gp="mc",
#                                  seed_points=100,
#                                  project_name="material_data",
#                                  target=["yield strength (MPa)"],
#                                  target_mask=[True],
#                                  device="cpu",
#                                  n_iter=20)
# saasbo = SAASBO(saasbo_parameters=saasbo_params)
# saasbo.run_optimization()


