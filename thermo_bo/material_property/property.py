# Author: Osman Mamun

"""
Property computation function.
"""

import numpy as np 
from tc_python import TCPython, CompositionUnit
from typing import Dict, NamedTuple, Union

class AlloyData(NamedTuple):
    """
    Data class for alloy properties.
    
    density: density of the alloy
    electrical_conductivity: electircal conductiveity of the alloy
    heat_capacity: heat capacity of the alloy
    electric_resistivity: electrical resistivity of the alloy
    thermal_conductivity: thermal conductivity of the alloy
    thermal_diffusivity: thermal diffusivity of the alloy
    thermal_resistivity: thermal resistivity of the alloy
    linear_thermal_expansion: linear thermal expansion of the alloy
    """
    density: Union[float, np.nan]
    electrical_conductivety: Union[float, np.nan]
    heat_capacity: Union[float, np.nan]
    electric_resistivity: Union[float, np.nan]
    thermal_conductiveity: Union[float, np.nan]
    thermal_diffusivity: Union[float, np.nan]
    thermal_resistivity: Union[float, np.nan]
    linear_thermal_expansion: Union[float, np.nan]

def compute_property(params: Dict[str, float]) -> AlloyData:
    """
    A function to compute density using thermocalc.

    :param params: a dictionary containing materials composition and temperature.
    :param property: which property to compute.
    """
    
    assert set(params.keys()).issubset({'Al', 'Ag', 'B', 'Be', 'Bi', 'C', 'Ca', 
        'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Fe', 'Ga', 'Ge', 'H', 'Hf', 'In', 'K', 
        'La', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pr', 'S', 
        'Sc', 'Se', 'Si', 'Sn', 'Sr', 'Te', 'Ti', 'V', 'Y', 'Zn', 'Zr', 'Temp'}), "element not supported!"

    
    elements = [i for i in params.keys() if i != 'Temp']
    with TCPython() as session:
        system = (session.select_database_and_elements("TCAL8", elements).get_system())
        model_path = "/home/ebnahaib/ThermoBO/thermo_bo/material_property/property_models"
        calculation = system.with_property_model_calculation("Simplified equilibrium with freeze-in temperature", model_path)
        calculation.set_composition_unit(CompositionUnit.MASS_PERCENT)
        for k, v in params.items():
            if k.lower() in ['al', 'temp']:
                continue
            calculation.set_composition(k, v)
        freeze_in_celsius = 350
        calculation.set_argument('Freeze-in-temperature', freeze_in_celsius + 273.15)
        calculation.set_temperature(params['Temp'] + 273.15)
        result = calculation.calculate()
        return AlloyData(result.get_value_of('Density (g/cm3)'),
                         result.get_value_of('Electric conductivity (S/m)'),
                         result.get_value_of('Heat capacity (J/(mol K))'),
                         result.get_value_of('Electric resistivity (ohm m)'),
                         result.get_value_of('Thermal conductivity (W/(mK))'),
                         result.get_value_of('Thermal diffusivity (m2/s)'),
                         result.get_value_of('Thermal resistivity (mK/W)'),
                         result.get_value_of('Linear thermal expansion (1/K)'))


