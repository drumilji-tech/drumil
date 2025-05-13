from enum import Enum, auto


class AggTypes(Enum):
    AVERAGE = "mean"
    STD = "std"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    SUM = "sum"


class DataSourceType(Enum):
    CSV = auto()
    TXT = auto()
    EXCEL = auto()
    MS_SQL_DATABASE = auto()
    POSTGRES_DATABASE = auto()
    INTERNAL_CALCULATED = auto()


# these values will become property names
# they normalize the different ways these show
# up in the tag names for different sites
# be aware that wehn adding or changing these the application needs
# to be able to map pi tags to names
# and names to pi tags
class ComponentTypes(Enum):
    ACTIVE_POWER = "Active_Power"
    CONTROL_GROUND = "Ctrl_Gnd_Temp"
    CONTROL_HUB = "Ctrl_Hub_Temp"
    CONTROL_TOP = "Ctrl_Top_Temp"
    CONTROL_VCP = "Ctrl_VCP_Temp"
    EFFICIENCY = "Efficiency"
    EXPECTED_POWER = "Expected_Power"
    FAULT_CODE = "Fault_Code"
    GEARBOX_COOLING_WATER_TEMP = "Gbx_CoolingWater_Temp"
    GEARBOX_MAIN_TANK = "Gbx_Internal_Temp"
    GEARBOX_OIL = "Gbx_Oil_Temp"
    GENERATOR_1_TEMPERATURE = "Gen_Internal_Temp"
    GENERATOR_2_TEMPERATURE = "Gen_Internal_Temp"
    GENERATOR_BEARING_DRIVE_END = "Gen_Brg_DE_Temp"
    GENERATOR_BEARING_NON_DRIVE_END = "Gen_Brg_NDE_Temp"
    GENERATOR_COOLING = "Gen_CoolingFluid_Temp"
    GENERATOR_SLIP_RING = "Gen_Slip_Ring_Temp"
    GEN_SPEED = "Gen_Speed"
    GENERATOR_WINDING_PHASE_A = "Gen_Windings_PhaseA_Temp"
    GENERATOR_WINDING_PHASE_B = "Gen_Windings_PhaseB_Temp"
    GENERATOR_WINDING_PHASE_C = "Gen_Windings_PhaseC_Temp"
    HIGH_SPEED_BEARING = "Gbx_Brg_HighSpd_Temp"
    HUB = "Hub_Temp"
    HYDRAULIC_OIL_TEMP = "HPU_Oil_Temp"
    INTERMEDIATE_SHAFT_BEARING_GE = "Gbx_Brg_IntrSpd_NDE_Temp"
    INTERMEDIATE_SHAFT_BEARING_TE = "Gbx_Brg_IntrSpd_DE_Temp"
    LOST_ENERGY = "Lost_Energy"
    LOST_REVENUE = "Lost_Revenue"
    MAIN_BEARING = "Main_Brg_Temp"
    NACELLE = "Nacelle_Temp"
    NACELLE_WIND_DIRECTION = "Nacl_Wind_Dir"
    NACELLE_WIND_SPEED = "Nacl_Wind_Speed"
    NACELLE_ESTIMATED_WIND_SPEED = "Nacl_Est_Wind_Speed"
    NACELLE_AD_ADJ_WIND_SPEED = "Nacl_AD_ADj_Wind_Speed"
    OPERATING_STATE = "Op_state"
    PITCH_ANGLE_A = "Blade_pitch_A"
    PITCH_ANGLE_B = "Blade_pitch_B"
    PITCH_ANGLE_C = "Blade_pitch_C"
    PITCH_REF = "Blade_pitch_Ref"
    ROTOR_PHASE_A = "Rtr_PhaseA_Temp"
    ROTOR_PHASE_B = "Rtr_PhaseB_Temp"
    ROTOR_PHASE_C = "Rtr_PhaseC_Temp"
    SEVERITY = "Relative Deviation"
    SPINNER = "Hub_Temp"
    TURBINE_STATE_SCADA = "turbine_state_scada"
    TRANSFORMER_AUX = "Xfrm_Aux_Temp"
    TRANSFORMER_CORE_PHASE_A = "Xfrm_MV_CorePhaseA_Temp"
    TRANSFORMER_CORE_PHASE_B = "Xfrm_MV_CorePhaseB_Temp"
    TRANSFORMER_CORE_PHASE_C = "Xfrm_MV_CorePhaseC_Temp"
    TRANSFORMER_PHASE_A = "Xfrm_MV_PhaseA_Temp"
    TRANSFORMER_PHASE_B = "Xfrm_MV_PhaseB_Temp"
    TRANSFORMER_PHASE_C = "Xfrm_MV_PhaseC_Temp"
    TRANSFORMER_HIGH_SPEED_WINDING_1 = "Xfmr_Winding1_Temp"
    TRANSFORMER_HIGH_SPEED_WINDING_2 = "Xfmr_Winding2_Temp"
    YAW_ERROR = "Yaw_Error"
    YAW_POSITION = "Yaw_Position"
