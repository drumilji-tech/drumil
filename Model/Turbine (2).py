from Model.Component import Component


class Turbine:
    """A wind turbine."""

    def __init__(self, name, data=None):
        """Initializes a new instance of the Turbine class.

        Args:
            name (str): The name of the turbine.
            data (any): The data associated with the turbine.
        """
        self.name = name
        self.components = {}
        self.data = data

        # for component_type in self.get_component_types():
        #     self._components[component_type] = Component(component_type)

        # Public properties for each component type
        # for component_type, component_obj in self._components.items():
        #     setattr(
        #         Turbine,
        #         component_type,
        #         property(
        #             lambda self, component_type=component_type: self._get_component(
        #                 component_type
        #             )
        #         ),
        #     )

        # def _get_component(self, component_type):
        """Gets the component of the specified type.

        Args:
            component_type (str): The type of the component.

        Returns:
            Component: The component of the specified type.
        """
        # return self._components[component_type]

    def get_data(self, start=None, end=None, freq="10T"):
        """Gets the data for the specified time range and frequency.

        Args:
            start (Optional[Union[str, datetime.datetime]]): The start time of the data range. Defaults to None.
            end (Optional[Union[str, datetime.datetime]]): The end time of the data range. Defaults to None.
            freq (Optional[str]): The frequency of the data. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the specified time range and frequency.
        """
        if self.data is None:
            raise ValueError("No data associated with turbine")

        if start is None:
            start = self.data.index[0]
        if end is None:
            end = self.data.index[-1]

        return self.data.loc[start:end].resample(freq).mean()

    def __str__(self):
        """Returns a string representation of the turbine.

        Returns:
            str: The string representation of the turbine.
        """
        return f"Turbine: {self.name}"
