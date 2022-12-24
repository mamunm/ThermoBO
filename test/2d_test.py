import sys
sys.path.append("/home/osman/git_repos/ThermoBO")

import torch
import numpy as np
from thermo_bo.data import get_data, DataParameters
from thermo_bo.models import ExactGPModel
from thermo_bo.saasbo import SAASBOParameters, SAASBO
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch import fit_gpytorch_model
import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.sampling.samplers import SobolQMCNormalSampler

df, properties = get_data(DataParameters("computational", ["Density (g/cm3)",
                        "Density (g/cm3)"]))
xs = torch.tensor(df[properties].to_numpy())
ys = torch.tensor(df[["Density (g/cm3)"]].to_numpy())
bounds = torch.stack([xs.min(0)[0], xs.max(0)[0]])

# plot the points
# plt.scatter(np.arange(len(ys.view(-1))), ys.view(-1))
# plt.axhline(500, c="r", linewidth=2)
# plt.xlabel("# data points")
# plt.ylabel("yield_strength")
# plt.grid()
# plt.show()

NUM_RESTARTS =  10
RAW_SAMPLES = 1024
MC_SAMPLES = 256

def initialize_model(train_x, train_y):
    
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:, i]
        models.append(
            SingleTaskGP(train_x, train_objective.unsqueeze(-1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def generate_next_candidate(x, y, x_test, n_candidates=1):
    
    mll, model = initialize_model(x, y)
    fit_gpytorch_model(mll)

    sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

    train_x = normalize(x, bounds)
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean
    
    acq_fun_list = []
    for _ in range(n_candidates):
        
        weights = sample_simplex(1).view(-1)
        objective = GenericMCObjective(
            get_chebyshev_scalarization(
                weights,
                pred
            )
        )
        acq_fun = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            sampler=sampler,
            X_baseline=train_x,
            prune_baseline=True,
        )
        acq_fun_list.append(acq_fun)
    

    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_fun_list[0],
        choices=normalize(x_test, bounds), 
        q=n_candidates, unique=True
    )
    print(candidates)

    return unnormalize(candidates, bounds)

def plot_candidates(candidates):
    plt.scatter(np.arange(len(ys.view(-1))), ys.view(-1))
    plt.scatter(np.arange(len(candidates.view(-1))), candidates.view(-1), c="r")
    #plt.axhline(500, c="r", linewidth=2)
    plt.xlabel("# data points")
    plt.ylabel("Density (g/cm3)")
    plt.grid()
    plt.show()
    
def get_loc(candidates, xx_test):
    loc = []
    for i, xx in enumerate(xx_test):
        for cand in candidates:
            if np.allclose(xx, cand):
                loc.append(i)
                break
    return np.array(loc)

n_iter = 2
n_start = 3
n_samples = 1

seed = np.random.choice(len(ys), n_start, replace=False)
XX = xs[seed]
YY = ys[seed]

XX_test = np.delete(xs, seed, 0)
YY_test = np.delete(ys, seed, 0)
print(YY_test)

for i in range(n_iter):
    print(f"Iteration {i}")

    candidates = generate_next_candidate(XX, YY, XX_test, n_candidates=n_samples)

    #print(f"Candidates: {candidates}")
    loc = get_loc(candidates, XX_test)
    #print(loc)
    print(XX_test[loc], YY_test[loc])
    anti_loc = np.delete(np.arange(len(YY_test)), loc, 0)

    XX = torch.cat([XX, candidates])
    YY = torch.cat([YY, YY_test[loc]], dim=0)
    plot_candidates(YY_test[loc])
    XX_test = XX_test[anti_loc]
    YY_test = YY_test[anti_loc]

    