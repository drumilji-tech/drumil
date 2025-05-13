from Utils.Enums import AggTypes


class DataAttribute:
    """
    responsible for storing different versions of a component aggregate
    ex ----> Turbine.Component.DataAttribute.[property]
             might be WAK_T001.main_brg.mean.clean
    """

    def __init__(self, name):
        self.column_name = name
        self.clean = None
        self.raw = None
        self.gradient = None
        self.range = None
        self.online = None
        self.flagged = None

    def __str__(self):
        return f"DataAttribute: {self.name}"
