from tc_python import *


class LinearInTPythonModel(PropertyModel):
    """
    Jython Property Model representing a linear equation. The model is very straightforward and primarily
    focused on the linear equation.

    All of the methods are required to be implemented in order to fulfill the contract with the Property Model
    Framework (except 'provide_model_parameters()' and 'set_model_parameters()' which are optional). Consult the
    'Python Property Model Framework' documentation for details.
    """

    def __init__(self, locale="en-US"):
        super().__init__(locale)
        # the model parameters (here 'a' and 'b') can be represented internally in any way, here we use a dict
        self._model_parameters = dict()
        self._model_parameters["a"] = 900.0
        self._model_parameters["b"] = 1.0

    def provide_model_category(self):
        """Called by Thermo-Calc when the model should provide its category (shown in the Thermo-Calc model tree)."""
        return ["Examples"]

    def provide_model_name(self):
        """Called by Thermo-Calc when the model should provide its name (shown in the Thermo-Calc model tree)."""
        return "Simple linear model"

    def provide_model_description(self):
        """Called by Thermo-Calc when the model should provide its detailed description."""
        description = "This is an parameterized simple linear model."
        return description

    def provide_calculation_result_quantities(self):
        """Called by Thermo-Calc when the model should provide its result quantity objects."""
        # in TC-Python this is called an 'argument' with the argument id 'result'
        result_quantities = [create_general_quantity("result", "result")]
        return result_quantities

    def provide_model_parameters(self):
        """Called by Thermo-Calc when the model should provide its current model parameter values."""
        # the required return type is 'dict(str, float)' (such as {"a": 900, "b": 1}), we already store the model
        # parameters in that format
        return self._model_parameters

    def set_model_parameter(self, parameter_id, value):
        """Called by Thermo-Calc when a model parameter should be reset during the optimization."""
        # this will actually update the model parameter for the next 'evalulate_model()' call
        self._model_parameters[parameter_id] = value

    def provide_ui_panel_components(self):
        """Called by Thermo-Calc when the model should provide its UI components for the model panel to be plotted."""
        ui_components = []
        return ui_components

    def evaluate_model(self, context: CalculationContext):
        """
        Called by Thermo-Calc when the model should be actually calculated.

        .. note::

            Once the fitting is done, in the final model one would replace the model parameters by the optimum fit
            values.
        """
        T = context.get_temperature()
        # our linear model equation using model parameters that still need to be fitted to the 'experimental' data
        result = self._model_parameters["a"] + self._model_parameters["b"] * T

        context.set_result_quantity_value("result", result)
