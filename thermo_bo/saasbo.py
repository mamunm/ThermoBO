# Author: Osman Mamun

"""
SAASBO class.
"""

import contextlib
import sys
from typing import Dict, List, Literal, NamedTuple, Tuple

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    DominatedPartitioning, NondominatedPartitioning)
from botorch.utils.multi_objective.scalarization import \
    get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from .data import DataParameters, get_data
from .models import ExactGPModel, TorchGP, SaasGP


class ThermoCalcFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = ThermoCalcFile()
    yield
    sys.stdout = save_stdout

class SAASBOParameters(NamedTuple):
    """
    Input parameters for SAASBO module.
    :param gp: Gaussian process
    :param sampler: sampling strategy
    :param partitioning: partitioning strategy
    :param acq_strategy: acquisition strategy
    :param seed_points: number of initial seed points
    :param project_name: name of the project
    :param targets: target properties to optimize
    :param targets_mask: boolean to inidicate target to be maximized (True) or minimized (False)
    :param device: which device to use (cpu or gpu)
    :param n_iter: number of iterations to perform
    :param cand_size: number of candidates to acquire in each iteration
    :param exact_gp_n_iter: number of iterations for exact gp optimization
    """
    gp: Literal["exact", "torch"]
    acq_startegy: Literal["q", "q_noisy"]
    sampler: Literal["iid", "sobol"]
    partitioning: Literal["dominated", "non-dominated"]
    seed_points: int
    project_name: str
    targets: List[str]
    targets_mask: List[bool]
    device: str
    n_iter: int = 10
    cand_size: int = 4
    exact_gp_n_iter: int = 1000
    
    
class SAASBO:
    """
    Bayesian optimization for ThermoCalc property optimization.
    """
    def __init__(self, saasbo_parameters: SAASBOParameters) -> None:
        """
        Initialize the ThermoBO class with the user defined parameters.

        Args:
            thermobo_parameters (ThermoBOParameters): input parameters to the class.
        """
        self._inputs = saasbo_parameters
        
        assert self._inputs.device in ("cpu", "cuda"), f"device must be cpu or cuda but found {self._inputs.device}" 
        assert isinstance(saasbo_parameters.targets, list), f"Target must be a list of targets!"
        assert self._inputs.gp in ("exact", "torch", "saas"), f"gp can be exact or torch but found {self._inputs.gp}"
        
        if self._inputs.gp == "torch":
            self.gp = TorchGP()
        elif self._inputs.gp == "exact":
            self.gp = ExactGPModel(self._inputs.exact_gp_n_iter)
        elif self._inputs.gp == "saas":
            self.gp = SaasGP()
        else:
            raise ValueError("gp not recognized. Only torch and exact are allowed.")
        self.df, self.features = get_data(DataParameters(self._inputs.project_name, 
            self._inputs.targets))
        for target in self._inputs.targets:
            self.df[f"saasbo_{target}"] = np.nan
        self.acquire_seed_points()
    
    def acquire_seed_points(self) -> None:
        """
        function to acquire seed points.
        """
        seed_points = np.random.choice(np.arange(len(self.df)), self._inputs.seed_points, 
            replace=False)
        for target in self._inputs.targets:
            self.df.loc[seed_points, f"saasbo_{target}"] = self.df.loc[seed_points, target]

    def save_df(self, path: str) -> None:
        """
        Save the df as a csv file.
        
        :param path: path to save the file
        """
        self.df.to_csv(f"{path}.csv", index=False)

    def get_iloc(self, data: Dict[str, float]) -> None:
        """
        Get row index from property data.
        """
        for idx in range(len(self.df)):
            row_dict = self.df.iloc[idx].to_dict()
            if all(np.isclose(data[feat], row_dict[feat]) for feat in self.features):
                return idx
    
    def get_train_test_data(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get train test tensor from the dataframe.

        Returns:
            A tuple of three tensors representing train and test set.
        """
        temp_df = self.df[~np.isnan(self.df[f"saasbo_{self._inputs.targets[0]}"])]
        X = torch.tensor(
                temp_df[self.features].to_numpy(), 
                dtype=torch.float64, device=torch.device(self._inputs.device))
        Y = torch.tensor(
                temp_df[self._inputs.targets].to_numpy(),
                dtype=torch.float64, device=torch.device(self._inputs.device))
        temp_df = self.df[np.isnan(self.df[f"saasbo_{self._inputs.targets[0]}"])]
        test_X = torch.tensor(
                temp_df[self.features].to_numpy(),
                dtype=torch.float64, device=torch.device(self._inputs.device))
        return X, Y, test_X
    
    def generate_next_candidates(self, X: Tensor, Y: Tensor, X_test: Tensor) -> Tensor:
        """Generate next candidates based on the startegy

        Args:
            X (Tensor): Feature tensor
            Y (Tensor): Target tensor
            X_test (Tensor): Test tensor
        """
        # bounds = torch.stack([X.min(0)[0], X.max(0)[0]])
        # X = normalize(X, bounds)
        # X_test = normalize(X_test, bounds)
        scaler = StandardScaler()
        X = torch.tensor(scaler.fit_transform(X), dtype=torch.float64,
            device=torch.device(self._inputs.device))
        X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float64,
            device=torch.device(self._inputs.device))
        gp = self.gp.get_optimized_gp(X, Y, self._inputs.targets_mask)
        
        if self._inputs.sampler == "sobol":
            sampler = SobolQMCNormalSampler(num_samples=1024, collapse_batch_dims=False)
        elif self._inputs.sampler == "iid":
            sampler = IIDNormalSampler(num_samples=1024, collapse_batch_dims=False)
        else:
            raise ValueError("sampler strategy not recognized. Only iid and sobor are allowed.")
        
        if Y.shape[-1] == 1:
            with torch.no_grad():
                pred = gp.posterior(X).mean
            weights = sample_simplex(1).view(-1)
            objective = GenericMCObjective(get_chebyshev_scalarization(weights,
                pred))
            if self._inputs.acq_startegy == "q":
                acq_fun = qExpectedImprovement(model=gp, objective=objective,
                    sampler=sampler, best_f=max(Y) if self._inputs.targets_mask[0] else max(-Y))
            elif self._inputs.acq_startegy == "qnoisy":
                acq_fun = qNoisyExpectedImprovement(model=gp, objective=objective, 
                    sampler=sampler, X_baseline=X, prune_baseline=True)
            else:
                raise ValueError("acquisition strategy not recognized. Only q and qnoisy are allowed.")
        else:
            reference_points = Y.min(0)[0]
            Y_train = Y.clone().detach()
            for i in range(Y.shape[-1]):
                reference_points[i] = reference_points[i] if self._inputs.targets_mask[i] else -reference_points[i]
                Y_train[:, i] = Y_train[:, i] if self._inputs.targets_mask[i] else -Y_train[:, i]
            if self._inputs.partitioning == "dominated":
                partitioning = DominatedPartitioning(ref_point=reference_points, Y=Y_train)
            elif self._inputs.partitioning == "non-dominated":
                partitioning = NondominatedPartitioning(ref_point=reference_points, Y=Y_train)
            else:
                raise ValueError("partitioning not recognized. Only dominated and non-dominated are allowed.")
            if self._inputs.acq_startegy == "q":
                acq_fun = qExpectedHypervolumeImprovement(model=gp, sampler=sampler,
                    ref_point=reference_points, partitioning=partitioning)
            elif self._inputs.acq_startegy == "q_noisy":
                acq_fun = qNoisyExpectedHypervolumeImprovement(model=gp, sampler=sampler,
                    ref_point=reference_points, partitioning=partitioning, incremental_nehvi=True, 
                    X_baseline=X, prune_baseline=False,)
            else:
                raise ValueError("acquisition strategy not recognized. Only q and qnoisy are allowed.")
        candidates, acq_values = optimize_acqf_discrete(acq_function=acq_fun,
            choices=X_test, q=self._inputs.cand_size, unique=True)
        # return unnormalize(candidates, bounds), acq_values
        candidates = torch.tensor(scaler.inverse_transform(candidates),
            dtype=torch.float64, device=torch.device(self._inputs.device))
        return candidates, acq_values
        
    def run_optimization(self) -> None:
        """
        Construct the individual GP models from model parameters.
        """
        logger.info(f"Number of total data points: {len(self.df)}")
        for i in range(1, self._inputs.n_iter+1):
            logger.info(f"Iteration {i}/{self._inputs.n_iter}")
            X, Y, X_test = self.get_train_test_data()
            candidates, _ = self.generate_next_candidates(X, Y, X_test)
            logger.info(f"Collenting candidates property...")
            for cand in candidates:
                idx = self.get_iloc(dict(zip(self.features, cand)))
                for target in self._inputs.targets:
                    logger.info(f"Collected candidate property: {self.df.at[idx, target]}")
                    self.df.at[idx, f"saasbo_{target}"] = self.df.at[idx, target]
            logger.info("Best see so far:")
            for target, mask in zip(self._inputs.targets, self._inputs.targets_mask):
                global_f = self.df[target].max() if mask else self.df[target].min()
                best_f = self.df[f"saasbo_{target}"].max() if mask else self.df[f"saasbo_{target}"].min()
                logger.info(f"{target}: {best_f} {global_f}")
        