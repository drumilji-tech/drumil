import unittest

from Model.Component import Component
from Utils.Enums import AggTypes, ComponentTypes


class TestComponent(unittest.TestCase):
    def setUp(self):
        self.component = Component(ComponentTypes.GENERATOR_BEARING_DRIVE_END.value)

    def test_name_property(self):
        self.assertEqual(
            self.component.name, ComponentTypes.GENERATOR_BEARING_DRIVE_END.value
        )

    def test_aggregation_properties(self):
        for agg_type in AggTypes:
            prop_name = agg_type.value
            self.assertTrue(hasattr(self.component, prop_name))
            prop = getattr(self.component, prop_name)
            self.assertEqual(prop, self.component._data_attributes[prop_name])
