"""Parameters for Online Filtering."""

from Utils.Enums import ComponentTypes

ONLINE_FILTER_PARAMETERS = {
    "BR2": {
        "GE_2_72_116": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2820},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        },
        "GE_2_82_127": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2970},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        },
    },
    "BTH": {
        "GE_2_3_116": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2450},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        }
    },
    "CFW": {
        "V126_3_45": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 660, "upper_bound": 1700},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 3500},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        }
    },
    "DHW": {
        "GE_2_82_127": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2870},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        },
        "GE_2_3_116": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        },
    },
    "GSW": {
        "V150_4_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 660, "upper_bound": 1700},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 4350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        },
        "V110_2_2": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 720, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2290},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        },
    },
    "GNT": {
        "SWT_2_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 625, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [0, 1]},
        }
    },
    "GNP": {
        "SWT_2_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 625, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [0, 1]},
        }
    },
    "KAY": {
        "SWT_2_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 625, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [0, 1]},
        }
    },
    "PDK": {
        "V112_3_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 695, "upper_bound": 1650},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 3350},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        }
    },
    "RDG": {
        "G132_3_47": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 835, "upper_bound": 1500},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 3530},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [75, 100]},
        },
        "S108_2_42": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 640, "upper_bound": 1650},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2470},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [0, 1]},
        },
    },
    "SFK": {
        "V100_2": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 790, "upper_bound": 1500},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2050},
            ComponentTypes.PITCH_REF.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        }
    },
    "SKO": {
        "V136_3_6": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 870, "upper_bound": 1700},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 3650},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        }
    },
    "TBF": {
        "SWT_2_3": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 625, "upper_bound": 1550},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 2500},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [0, 1]},
        }
    },
    "WAK": {
        "GE_1_79_103": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 825, "upper_bound": 1500},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 1830},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [2]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [16, 30, 602]},
        }
    },
    "WHE": {
        "V136_3_45": {
            ComponentTypes.GEN_SPEED.value: {"lower_bound": 670, "upper_bound": 1600},
            ComponentTypes.ACTIVE_POWER.value: {"lower_bound": 5, "upper_bound": 3500},
            ComponentTypes.PITCH_ANGLE_A.value: {"lower_bound": -5, "upper_bound": 50},
            ComponentTypes.FAULT_CODE.value: {"normal_codes": [0]},
            ComponentTypes.OPERATING_STATE.value: {"normal_codes": [3]},
        }
    },
}
