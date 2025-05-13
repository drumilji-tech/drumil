"""Paremters related to Trips and Non-Trips."""

ge_normal_fault_codes = [0, 1, 2, 3]
vestas_normal_fault_codes = [0]
seimens_normal_fault_codes = [0]

NON_TRIP_CODES = {
    "BR2": ge_normal_fault_codes,
    "BTH": ge_normal_fault_codes,
    "CFW": vestas_normal_fault_codes,
    "DHW": ge_normal_fault_codes,
    "GSW": vestas_normal_fault_codes,
    "GNT": seimens_normal_fault_codes,
    "GNP": seimens_normal_fault_codes,
    "KAY": seimens_normal_fault_codes,
    "PDK": vestas_normal_fault_codes,
    "RDG": seimens_normal_fault_codes,
    "SFK": vestas_normal_fault_codes,
    "SKO": vestas_normal_fault_codes,
    "TBF": seimens_normal_fault_codes,
    "WAK": ge_normal_fault_codes,
    "WHE": vestas_normal_fault_codes,
}
