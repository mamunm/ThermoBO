# Author: Osman Mamun

"""
Models.
"""

from typing import List

import gpytorch
import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.transforms import normalize
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.models import ExactGP
from loguru import logger
from torch import Tensor
from botorch.models.transforms import Standardize
from botorch import fit_fully_bayesian_model_nuts


class SimpleCustomGP(ExactGP, GPyTorchModel):
    """Build an exact gp model using RBF kernel. (try matern kernel later)"""
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        """Make forward pass through the network."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class ExactGPModel:
    """
    Exact GP class.
    """
    def __init__(self, n_iter: int) -> None:
        """_summary_

        Args:
            n_iter (int): number of iterations
        """
        self.n_iter = n_iter
    
    def get_optimized_gp(self, X: Tensor, y:Tensor, y_mask: List[bool], verbose: bool=False) -> gpytorch.models.GP:
        """
        Get an optimized exact gp.
        
        :param X: feature tensor
        :param y: target tensor
        :param y_mask: mask for targets
        :param verbose: wheather to print model training statistics 
        """
        # bounds = torch.stack([X.min(0)[0], X.max(0)[0]])
        # X = normalize(X, bounds)
        models = []
        for i in range(y.shape[-1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            train_objective = y[:, i] if y_mask[i] else -y[:, i]
            model = SimpleCustomGP(X, train_objective, likelihood)
            
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for i in range(self.n_iter):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, train_objective.unsqueeze(-1)).sum()
                loss.backward()
                if (i % 50 == 0) & verbose:
                    logger.info(f"Iter {i+1}/{self.n_iter+1} - Loss: {loss}   noise: {model.likelihood.noise.item()}")
                optimizer.step()
            model.eval()
            likelihood.eval()
            models.append(model)
        
        return ModelListGP(*models)


class TorchGP:
    """
    Torch GP class.
    """
    def __init__(self) -> None:
        """
        Initialize the TrochGP class
        """
        self.name = "TorchGP"
    
    def get_optimized_gp(self, X: Tensor, y: Tensor, y_mask: List[bool]) -> SaasFullyBayesianSingleTaskGP:
        """
        Get an optimized mc gp.
        
        :param X: feature tensor
        :param y: target tensor
        :param y_mask: mask for targets
        """
        # bounds = torch.stack([X.min(0)[0], X.max(0)[0]])
        # train_x = normalize(train_x, bounds)
        models = []
        for i in range(y.shape[-1]):
            train_objective = y[:, i] if y_mask[i] else -y[:, i]
            models.append(SingleTaskGP(X, train_objective.unsqueeze(-1)))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model
    
class SaasGP:
    """
    Saas GP class.
    """
    def __init__(self) -> None:
        """
        Initialize the TrochGP class
        """
        self.name = "SaasGP"
    
    def get_optimized_gp(self, X: Tensor, y: Tensor, y_mask: List[bool]) -> SaasFullyBayesianSingleTaskGP:
        """
        Get an optimized mc gp.
        
        :param X: feature tensor
        :param y: target tensor
        :param y_mask: mask for targets
        """
        # bounds = torch.stack([X.min(0)[0], X.max(0)[0]])
        # train_x = normalize(train_x, bounds)
        models = []
        for i in range(y.shape[-1]):
            train_objective = y[:, i] if y_mask[i] else -y[:, i]
            gp = SaasFullyBayesianSingleTaskGP(X, train_objective.unsqueeze(-1), 
                train_Yvar=torch.full_like(train_objective.unsqueeze(-1), 1e-6), outcome_transform=Standardize(m=1))
            fit_fully_bayesian_model_nuts(gp, warmup_steps=512, num_samples=256, 
                thinning=16, disable_progbar=True)
            models.append(gp)
        model = ModelListGP(*models)
        return model

    
