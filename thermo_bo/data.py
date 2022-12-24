# Author: Osman Mamun

"""
Data Collection class.
"""

from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Set

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataParameters(NamedTuple):
    project_name: Literal["experimental", "computational"]
    target: List[str]

EXPERIMENTAL_DATA_PROP: Set[str] = {
    "source", "source type", "V", "Mo", "Cu", 
    "Ag", "Zn", "Co", "Pb", "B", "Sn", "Cr", "Zr", "Sb", "Na", "Si", "Fe", 
    "Mg", "Mn", "Bi", "Be", "Ni", "Sr", "Ti", "Al", "Ga", "temper", 
    "holding temperature (°C)", "holding time (h)", "test temperature (°C)"
    }

EXPERIMENTAL_DATA_TARGET: Set[str] = {"yield strength (MPa)",
                                  "ultimate tensile strength (MPa)", 
                                  "elongation (%)",
                                  "yield strength min (MPa)", 
                                  "ultimate tensile strength min (MPa)", 
                                  "elongation min (%)", 
                                  "yield strength max (MPa)",
                                  "ultimate tensile strength max (MPa)", 
                                  "elongation max (%)"}

COMPUTATIONAL_DATA_PROP: Set[str] = {"Si", "Fe", "Cu", "Al", "Mg", "Na",
                                     "Ca", "P", "Sn", "Zn", "Cr", "Mn", "Zr",
                                     "Sc", "Ti", "Ag", "Li", "Ni", "Pb", "B",
                                     "La", "Ga", "In", "V", "Er", "Y", "Sr",
                                     "C", "Nb", "Be", "Bi", "Mo", "Cd", "Ce",
                                     "Ge", "Nd", "Pr", "K", "Hf", "S", "H",
                                     "Co", "temperature"}

COMPUTATIONAL_DATA_TARGET: Set[str] = {"Density (g/cm3)",
                                       "Electric conductivity (S/m)", 
                                       "Heat capacity (J/(mol K))",
                                       "Thermal conductivity (W/(mK))", 
                                       "Thermal diffusivity (m2/s)",
                                       "Linear thermal expansion (1/K)"}


DATA_SOURCES: Dict[str, Path] = {
    "experimental": Path(__file__).parents[1] / "data/experimental_data.csv",
    "computational": Path(__file__).parents[1] / "data/computational_data.csv"
    }

DATA_PROP: Dict[str, Set[str]] = {
     "experimental": EXPERIMENTAL_DATA_PROP,
     "computational": COMPUTATIONAL_DATA_PROP}

DATA_TARGET: Dict[str, Set[str]] = {
     "experimental": EXPERIMENTAL_DATA_TARGET,
     "computational": COMPUTATIONAL_DATA_TARGET
     }


def get_data(data_params: DataParameters) -> pd.DataFrame:
    """
    Get the appropriate data as a dataframe.

    Args:
        data_params (DataParameters): input parameters to the class
    """
    assert data_params.project_name in {"experimental", "computational"}, "The requested data is not available!"
    assert set(data_params.target).issubset(DATA_TARGET[data_params.project_name]), "One or many requested target is not available!"
    
    df = pd.read_csv(DATA_SOURCES[data_params.project_name], low_memory=False)
    df.dropna(subset=data_params.target, inplace=True)
    if data_params.project_name == "experimental":
        df["temper"] = LabelEncoder().fit_transform(df["temper"].to_numpy())
    non_chem_props = ["source", "source type", "holding temperature (°C)", 
                      "holding time (h)", "test temperature (°C)", "temperature"]
    features = [i for i in DATA_PROP[data_params.project_name] if i not in non_chem_props]
    df.fillna(dict(zip(features, [0]*len(features))), inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    
    for prop in non_chem_props:
        if prop in df.columns:
            df = df[df[prop].notna()]
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() < 3])
    properties = [i for i in DATA_PROP[data_params.project_name] if i in df.columns]
    df = df.reset_index(drop=True)
    return df, properties


# class ThermoCalcFile(object):
#     def write(self, x): pass

# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = ThermoCalcFile()
#     yield
#     sys.stdout = save_stdout

# class DataCollectionParameters(NamedTuple):
#     """
#     Input parameters for data module.
#     :param search_space: a dictionary or csv defining the search space
#     """
#     search_space: Dict[str, Tuple[float, float, int]]

# class DataCollection:
#     """
#     Thermocalc data collection class.
#     """
#     def __init__(self, data_parameters: DataParameters) -> None:
#         """
#         Initialize the DataCollection class with the user defined parameters.

#         Args:
#             data_parameters (DataParameters): input parameters to the class.
#         """
#         self._inputs = data_parameters
#         self.property_fucntion = compute_property
#         self.construct_search_space()
         
#     def construct_search_space(self) -> None:
#         """
#         Construct the search space from a dict.
#         """
#         logger.info("Constructing the search space for 1D optimization.")
#         self.df = pd.DataFrame([list(j) for j in product(*[np.linspace(*v) 
#             for v in self._inputs.search_space.values()])], 
#             columns=self._inputs.search_space.keys())
#         self.df["Al"] = self.df.apply(lambda x: 100 - sum(x[i] for i in self.df.columns if i != 'Temp'), axis=1)
#         self.df[self._inputs.target] = np.nan
        
#     def collect_data(self) -> None:
#         """
#         Collect data.
#         """
#         for idx in range(len(self.df)):
#             with nostdout():
#                 prop = self.property_function(dict(zip(self.features, cand)))
#                 idx = self.get_iloc(dict(zip(self.features, cand)))
#                 if not np.isnan(prop):
#                     self.df.at[idx, self._inputs.target] = prop
                    