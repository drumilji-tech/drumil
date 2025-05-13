"""Constants for the Range Filter."""

from Utils.Enums import ComponentTypes


default_direction_range_params = {
    "lower_bound": 0,
    "upper_bound": 360,
    "invalid_flag": -9999,
}
default_yaw_error_range_params = {
    "lower_bound": -180,
    "upper_bound": 180,
    "invalid_flag": -9999,
}
default_wind_speed_range_params = {
    "lower_bound": 0,
    "upper_bound": 50,
    "invalid_flag": -9999,
}
RANGE_FILTER_PARAMETERS = {
    "BR2": {
        "GE_2_72_116": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
        "GE_2_82_127": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
    },
    "BTH": {
        "GE_2_3_116": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "CFW": {
        "V126_3_45": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "DHW": {
        "GE_2_82_127": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
        "GE_2_3_116": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
    },
    "GSW": {
        "V150_4_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 5000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 5000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
        "V110_2_2": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
    },
    "GNT": {
        "SWT_2_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "GNP": {
        "SWT_2_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "KAY": {
        "SWT_2_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "PDK": {
        "V112_3_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "RDG": {
        "G132_3_47": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
        "S108_2_42": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        },
    },
    "SFK": {
        "V100_2": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "SKO": {
        "V136_3_6": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "TBF": {
        "SWT_2_3": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "WAK": {
        "GE_1_79_103": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
    "WHE": {
        "V136_3_45": {
            ComponentTypes.ACTIVE_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_GROUND.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_TOP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.CONTROL_VCP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.EXPECTED_POWER.value: {
                "lower_bound": 5,
                "upper_bound": 4000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_COOLING_WATER_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_MAIN_TANK.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEARBOX_OIL.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_1_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_2_TEMPERATURE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_BEARING_NON_DRIVE_END.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_COOLING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_SLIP_RING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GEN_SPEED.value: {
                "lower_bound": 0,
                "upper_bound": 2000,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.GENERATOR_WINDING_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HIGH_SPEED_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HUB.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.HYDRAULIC_OIL_TEMP.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_GE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.INTERMEDIATE_SHAFT_BEARING_TE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.MAIN_BEARING.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.PITCH_ANGLE_A.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_B.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_ANGLE_C.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.PITCH_REF.value: {
                "lower_bound": -180,
                "upper_bound": 180,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.ROTOR_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_AUX.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_CORE_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_A.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_B.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_PHASE_C.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_1.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.TRANSFORMER_HIGH_SPEED_WINDING_2.value: {
                "lower_bound": -50,
                "upper_bound": 200,
                "invalid_flag": -9999,
            },
            ComponentTypes.NACELLE_WIND_DIRECTION.value: default_direction_range_params,
            ComponentTypes.NACELLE_ESTIMATED_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.NACELLE_WIND_SPEED.value: default_wind_speed_range_params,
            ComponentTypes.YAW_ERROR.value: default_yaw_error_range_params,
            ComponentTypes.YAW_POSITION.value: default_direction_range_params,
        }
    },
}
