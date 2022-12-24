from tc_python import *


class FreezeInEquilibriumPythonModel(PropertyModel):

    _FREEZE_IN_TEMPERATURE = 'Freeze-in-temperature'

    _RESULT_QUANTITY_ELRS = 'Electric resistivity (ohm m)'
    _RESULT_QUANTITY_ELCD = 'Electric conductivity (S/m)'
    _RESULT_QUANTITY_THCD = 'Thermal conductivity (W/(mK))'
    _RESULT_QUANTITY_THRS = 'Thermal resistivity (mK/W)'
    _RESULT_QUANTITY_THDF = 'Thermal diffusivity (m2/s)'
    _RESULT_QUANTITY_DENSITY = 'Density (g/cm3)'
    _RESULT_QUANTITY_LINEAR_THERMAL_EXPANSION = 'Linear thermal expansion (1/K)'
    _RESULT_HEAT_CAPACITY = 'Heat capacity (J/(mol K))'

    def provide_calculation_result_quantities(self) -> List[ResultQuantity]:

        results = [create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_ELRS,
                                           description='Electric resistivity (ohm m)'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_ELCD,
                                           description='Electric conductivity (S/m)'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THCD,
                                           description='Thermal conductivity (W/(mK))'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THRS,
                                           description='Thermal resistivity (mK/W)'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THDF,
                                           description='Thermal diffusivity (m2/s)'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_HEAT_CAPACITY,
                                           description='Heat capacity (J/(mol K))'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_DENSITY,
                                           description='Density (g/cm3)'),
                   create_general_quantity(quantity_id=FreezeInEquilibriumPythonModel._RESULT_QUANTITY_LINEAR_THERMAL_EXPANSION,
                                           description='Linear CTE (physical) (1/K)'),
                   ]
        return results

    def provide_model_category(self) -> List[str]:
        return ['General Models']

    def provide_model_name(self) -> str:
        return 'Simplified equilibrium with freeze-in temperature'

    def provide_model_description(self) -> str:
        return """
Calculates equilibrium at the freeze-in temperature and evaluate the properties at a \
different temperature. The assumption is that diffusion and phase transformations are \
negligible when changing from the freeze-in-temperature and, therefore, that the phase \
amounts and compositions of phases are kept at all other temperatures.

Input parameters:
- Freeze-in temperature: The temperature where the equilibrium is calculated.

The model evaluates the following properties for the system:
- Electric resistivity (ohm m)
- Electric conductivity (S/m)
- Thermal conductivity (W/(mK))
- Thermal resistivity (mK/W)
- Thermal diffusivity (m2/s)
- Heat capacity (J/(mol K))
- Density (g/cm3)
- Linear thermal expansion (1/K)
"""

    def provide_ui_panel_components(self) -> List[UIComponent]:
        ui_components = []
        freeze_in_temperature = create_temperature_ui_component(component_id=FreezeInEquilibriumPythonModel._FREEZE_IN_TEMPERATURE,
                                                                name='Freeze-in-temperature',
                                                                description='Freeze-in-temperature where the equilibrium is calculated',
                                                                initial_temp=350.0)
        ui_components.append(freeze_in_temperature)
        return ui_components

    def before_evaluations(self, context: CalculationContext):
        self.calculation = context.system.with_single_equilibrium_calculation()

    def evaluate_model(self, context: CalculationContext):
        self.logger.info('Calculation starts.')
        freeze_in_temperature = context.get_ui_temperature_value(FreezeInEquilibriumPythonModel._FREEZE_IN_TEMPERATURE)
        evaluation_temperature = context.get_temperature()

        try:
            self.calculation.set_condition('T', freeze_in_temperature)
            for element, value in context.get_mass_fractions().items():
                condition = 'W({})'.format(element)
                self.calculation.set_condition(condition, value)

            result_freeze_in_equilibrium = self.calculation.calculate()
            res = result_freeze_in_equilibrium.change_temperature(evaluation_temperature)

            elrs = res.get_value_of('ELRS')
            thrs = res.get_value_of('THRS')

            molar_volume = res.get_value_of('VM')
            molar_mass = res.get_value_of('BM')

            # The dot-derivatives (e.g. Cp = Hm.T) can't be used under freeze-in conditions because they include
            # the contributions from the change of amount of phase with changed temperature( NP(phase).T ).
            # The amount of phases are assumed to be fixed at evaluation temperature under freeze-in equilibrium.
            # Let's use numerical derivatives instead.
            delta_temperature_num_diff = 1e-3

            vm_upper_temperature = res.get_value_of('VM')
            hm_upper_temperature = res.get_value_of('HM')

            res = result_freeze_in_equilibrium.change_temperature(evaluation_temperature - delta_temperature_num_diff)
            vm_lower_temperature = res.get_value_of('VM')
            hm_lower_temperature = res.get_value_of('HM')

            vm_dot_temp = (vm_upper_temperature - vm_lower_temperature) / delta_temperature_num_diff
            heat_capacity = (hm_upper_temperature - hm_lower_temperature) / delta_temperature_num_diff

            # Derived quantities:
            linear_thermal_expansion = vm_dot_temp / (molar_volume * 3.0)
            thcd = 1.0 / thrs
            thdf = thcd * molar_volume / heat_capacity
            elcd = 1.0 / elrs
            density = molar_mass / molar_volume * 1e-6

            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_ELRS, elrs)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_ELCD, elcd)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THCD, thcd)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THRS, thrs)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_THDF, thdf)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_HEAT_CAPACITY, heat_capacity)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_DENSITY, density)
            context.set_result_quantity_value(FreezeInEquilibriumPythonModel._RESULT_QUANTITY_LINEAR_THERMAL_EXPANSION, linear_thermal_expansion)

        except Exception as e:
            self.logger.error('Exception in calculation:' + str(e))

        self.logger.info('Calculation is finished.')