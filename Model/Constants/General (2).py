"""General Constants used by the Model."""

from Utils.Enums import ComponentTypes


CLEANING_ORDER = [
    "gradient_filtered_data",
    "range_filtered_data",
    "online_filtered_data_map",
]
TURBINE_TECH = {
    "BR2": {"GE_2_72_116": 2720, "GE_2_82_127": 2820},
    "BTH": {"GE_2_3_116": 2300},
    "CFW": {"V126_3_45": 3450},
    "DHW": {"GE_2_3_116": 2300, "GE_2_82_127": 2820},
    "GSW": {"V150_4_3": 4300, "V110_2_2": 2200},
    "GNT": {"SWT_2_3": 2300},
    "GNP": {"SWT_2_3": 2300},
    "KAY": {"SWT_2_3": 2300},
    "PDK": {"SWT_2_3": 2300},
    "RDG": {"G132_3_47": 3470, "S108_2_42": 2420},
    "SFK": {"V100_2": 2000},
    "SKO": {"V136_3_6": 3600},
    "TBF": {"SWT_2_3": 2300},
    "WAK": {"GE_1_79_103": 1790},
    "WHE": {"V136_3_45": 3450},
}

COMPRESSED_COLUMN_TYPES = {
    "BR2": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "BTH": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "CFW": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "DHW": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "GNT": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "GNP": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "KAY": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "PDK": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "SFK": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "SKO": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "TBF": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "WAK": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "WHE": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "GSW": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
    "RDG": [ComponentTypes.FAULT_CODE.value, ComponentTypes.OPERATING_STATE.value],
}

DO_NOT_CALCULATE_SEVERITY = [
    ComponentTypes.ACTIVE_POWER.value,
    ComponentTypes.EXPECTED_POWER.value,
    ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value,
    ComponentTypes.FAULT_CODE.value,
    ComponentTypes.NACELLE_WIND_DIRECTION.value,
    ComponentTypes.YAW_POSITION.value,
    ComponentTypes.GEN_SPEED.value,
    ComponentTypes.PITCH_ANGLE_A.value,
    ComponentTypes.PITCH_ANGLE_B.value,
    ComponentTypes.PITCH_ANGLE_C.value,
    ComponentTypes.PITCH_REF.value,
    ComponentTypes.NACELLE_WIND_SPEED.value,
    ComponentTypes.OPERATING_STATE.value,
]
