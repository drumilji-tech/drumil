from Utils.Enums import AggTypes
from Model.DataAttribute import DataAttribute


class Component:
    """
    A class representing a wind turbine component.

    Attributes:
        name (str): The name of the wind turbine component.
        _data_attributes (dict): A dictionary containing the data values of the component.

    Properties:
        COUNT: A property representing the count of the component.
        MIN: A property representing the minimum value of the component.
        MAX: A property representing the maximum value of the component.
        AVERAGE: A property representing the average value of the component.
    """

    def __init__(self, name):
        """
        Initializes a new instance of the Component class.

        Args:
            name (str): The name of the wind turbine component.
        """
        self.name = name
        self._data_attributes = {}

        # add a property per aggregation in agg_types enum
        for agg_type in AggTypes:
            self._data_attributes[agg_type.value] = DataAttribute(name=agg_type.value)

        for data_attribute_type, data_attribute_obj in self._data_attributes.items():
            setattr(
                Component,
                data_attribute_type,
                property(
                    lambda self, data_attribute_type=data_attribute_type: self._get_data_attribute(
                        data_attribute_type
                    )
                ),
            )

    def _get_data_attribute(self, data_attribute_type):
        """
        Returns the data attribute object for the given type.

        Args:
            data_attribute_type (str): The type of the data attribute.

        Returns:
            DataAttribute: The data attribute object for the given type.
        """
        return self._data_attributes[data_attribute_type]

    def __str__(self):
        """
        Returns a string representation of the wind turbine component.

        Returns:
            str: The string representation of the wind turbine component.
        """
        return f"Component: {self.name}"
