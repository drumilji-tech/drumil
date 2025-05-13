import unittest

import numpy as np
import pandas as pd

from Utils.Transformers import join_fault_descriptions
from Utils.UiConstants import NULL_FAULT_DESCRIPTION


class TestJoinFaultDescriptions(unittest.TestCase):
    def setUp(self):
        self.mock_fault_code_lookup = pd.DataFrame(
            {
                "Code": [1, 2, 1],
                "Description": ["Fresh", "Lots of Pulp", "Granny Smith"],
                "Project": ["Orange", "Orange", "Apple"],
            }
        )
        self.data_frame = pd.DataFrame(
            {
                "Turbine": ["Apple-T001", "Apple-B002", "Orange-T002", "Orange-T002"],
                "CodeColumn": [1, 2, 1, 2],
            }
        )

    def test_join_fault_descriptions(self):
        result = join_fault_descriptions(
            data_frame=self.data_frame,
            fault_description_df=self.mock_fault_code_lookup,
            code_colname="CodeColumn",
        )
        expected = pd.DataFrame(
            {
                "Turbine": ["Apple-T001", "Apple-B002", "Orange-T002", "Orange-T002"],
                "CodeColumn": pd.array([1, 2, 1, 2], dtype="int32"),
                "Description": [
                    "Granny Smith",
                    NULL_FAULT_DESCRIPTION,
                    "Fresh",
                    "Lots of Pulp",
                ],
            }
        )
        pd.testing.assert_frame_equal(result, expected)
