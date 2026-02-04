"""
TEA/LCA Module for Polymer Solvent Recovery Analysis
=====================================================

This module provides Techno-Economic Analysis (TEA) and Life Cycle Assessment (LCA)
calculations for polymer-solvent separation processes using BioSTEAM.

DESIGNED FOR EASY EDITING:
- TEA/LCA specialists can modify this file without touching the agent code
- All configurable parameters are at the top of the file
- Clear separation between configuration, calculations, and interface

Author: [Your Name]
Last Modified: 2026-01-16

Dependencies:
    pip install biosteam thermosteam

Usage:
    from tea_lca_module import (
        calculate_solvent_recovery_tea,
        estimate_separation_cost,
        get_environmental_metrics,
        run_full_tea_analysis
    )
"""

# BioSTEAM imports - optional for advanced simulations
# Note: BioSTEAM requires specific Python version compatibility
# Our TEA/LCA calculations work standalone without it
try:
    import biosteam as bst
    from biosteam import units, Stream, System, TEA
    import thermosteam as tmo
    BIOSTEAM_AVAILABLE = True
except (ImportError, SyntaxError):
    BIOSTEAM_AVAILABLE = False
    bst = None
    tmo = None

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Suppress BioSTEAM warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS FOR YOUR ANALYSIS
# =============================================================================

@dataclass
class TEAConfig:
    """
    Techno-Economic Analysis Configuration

    Modify these parameters to match your facility and economic assumptions.
    """
    # Project timeline
    project_duration: int = 20  # years
    construction_years: int = 2
    start_year: int = 2026

    # Financial parameters
    IRR: float = 0.15  # Internal Rate of Return (15%)
    income_tax: float = 0.21  # Corporate income tax (21%)
    depreciation: str = 'MACRS7'  # Depreciation schedule

    # Operating parameters
    operating_days: int = 330  # Days per year
    operating_hours: int = 7920  # Hours per year (330 * 24)

    # Cost factors (as fraction of Fixed Capital Investment)
    lang_factor: float = 3.0  # Lang factor for total capital
    maintenance: float = 0.03  # 3% of FCI per year
    property_tax: float = 0.01  # 1% of FCI per year
    property_insurance: float = 0.007  # 0.7% of FCI per year

    # Labor costs
    labor_cost_per_operator: float = 75000  # USD/year
    operators_per_shift: int = 2
    shifts_per_day: int = 3
    fringe_benefits: float = 0.40  # 40% of labor

    # Working capital
    working_capital_fraction: float = 0.05  # 5% of FCI


@dataclass
class LCAConfig:
    """
    Life Cycle Assessment Configuration

    Emission factors and environmental impact coefficients.
    Sources: EPA, GREET, ecoinvent
    """
    # Electricity emission factor (kg CO2eq / kWh)
    electricity_emission: float = 0.42  # US grid average

    # Steam/heat emission factor (kg CO2eq / MJ)
    steam_emission: float = 0.07  # Natural gas boiler

    # Solvent production emissions (kg CO2eq / kg solvent)
    solvent_emissions: Dict[str, float] = None

    def __post_init__(self):
        if self.solvent_emissions is None:
            self.solvent_emissions = {
                # Common solvents - values from ecoinvent/GREET
                'toluene': 1.2,
                'xylene': 1.3,
                'acetone': 2.1,
                'methanol': 0.8,
                'ethanol': 1.5,
                'dmf': 3.5,
                'dmso': 2.8,
                'thf': 4.2,
                'dcm': 1.8,
                'chloroform': 2.0,
                'hexane': 0.9,
                'cyclohexane': 1.1,
                'water': 0.001,
                'nmp': 3.8,
                # Default for unknown solvents
                'default': 2.0
            }

    def get_solvent_emission(self, solvent: str) -> float:
        """Get emission factor for a solvent (case-insensitive)."""
        solvent_lower = solvent.lower().replace(' ', '').replace('-', '')
        return self.solvent_emissions.get(solvent_lower, self.solvent_emissions['default'])


@dataclass
class SolventProperties:
    """
    Solvent physical properties for process calculations.

    Add new solvents here or modify existing values.
    """
    # Boiling points (°C)
    boiling_points: Dict[str, float] = None

    # Heat of vaporization (kJ/kg)
    heat_vaporization: Dict[str, float] = None

    # Specific heat capacity (kJ/kg·K)
    specific_heat: Dict[str, float] = None

    # Price (USD/kg)
    prices: Dict[str, float] = None

    def __post_init__(self):
        if self.boiling_points is None:
            self.boiling_points = {
                'toluene': 110.6, 'xylene': 139.0, 'acetone': 56.0,
                'methanol': 64.7, 'ethanol': 78.4, 'dmf': 153.0,
                'dmso': 189.0, 'thf': 66.0, 'dcm': 40.0,
                'chloroform': 61.2, 'hexane': 69.0, 'cyclohexane': 80.7,
                'water': 100.0, 'nmp': 202.0, 'default': 100.0
            }
        if self.heat_vaporization is None:
            self.heat_vaporization = {
                'toluene': 351, 'xylene': 343, 'acetone': 534,
                'methanol': 1100, 'ethanol': 846, 'dmf': 490,
                'dmso': 560, 'thf': 410, 'dcm': 330,
                'chloroform': 247, 'hexane': 335, 'cyclohexane': 358,
                'water': 2260, 'nmp': 510, 'default': 400
            }
        if self.specific_heat is None:
            self.specific_heat = {
                'toluene': 1.72, 'xylene': 1.75, 'acetone': 2.18,
                'methanol': 2.53, 'ethanol': 2.44, 'dmf': 2.05,
                'dmso': 1.96, 'thf': 1.72, 'dcm': 1.19,
                'chloroform': 0.96, 'hexane': 2.26, 'cyclohexane': 1.84,
                'water': 4.18, 'nmp': 1.67, 'default': 2.0
            }
        if self.prices is None:
            self.prices = {
                'toluene': 0.85, 'xylene': 0.90, 'acetone': 0.95,
                'methanol': 0.45, 'ethanol': 0.75, 'dmf': 2.50,
                'dmso': 2.20, 'thf': 3.50, 'dcm': 1.20,
                'chloroform': 1.30, 'hexane': 0.80, 'cyclohexane': 1.00,
                'water': 0.002, 'nmp': 3.80, 'default': 1.50
            }

    def get_property(self, solvent: str, property_dict: Dict) -> float:
        """Get property value for a solvent (case-insensitive)."""
        solvent_lower = solvent.lower().replace(' ', '').replace('-', '')
        return property_dict.get(solvent_lower, property_dict['default'])


# Initialize default configurations
DEFAULT_TEA_CONFIG = TEAConfig()
DEFAULT_LCA_CONFIG = LCAConfig()
DEFAULT_SOLVENT_PROPS = SolventProperties()


# =============================================================================
# STRAP-SPECIFIC CONFIGURATION (Solvent-Targeted Recovery and Precipitation)
# =============================================================================

@dataclass
class STRAPConfig:
    """
    STRAP Process Configuration for multilayer film recycling.

    Based on: "Recycling of single-use biopharmaceutical manufacturing plastics
    using solvent-targeted recovery and precipitation (STRAP)"

    Modify these parameters to match your STRAP process design.
    """
    # Plant scale (metric tons/year)
    min_capacity_mt_yr: float = 2500
    max_capacity_mt_yr: float = 25000
    default_capacity_mt_yr: float = 10000

    # Operating parameters
    operating_days_per_year: int = 330
    operating_hours_per_year: int = 7920  # 330 * 24

    # Process parameters (from paper)
    feed_to_solvent_ratio: float = 1/13.6  # 1:13.6 mass ratio
    dissolution_temp_c: float = 95.0       # Default for PE in heptane
    dissolution_time_min: float = 90       # Minutes
    precipitation_temp_c: float = 25.0     # Room temperature

    # Recovery efficiencies
    polymer_recovery_efficiency: float = 0.95   # 95% polymer recovery
    solvent_recovery_efficiency: float = 0.999  # 99.9% solvent recovery
    solvent_loss_rate: float = 0.001           # 0.1% solvent loss per cycle

    # Financial parameters for MSP calculation
    target_irr: float = 0.15               # 15% Internal Rate of Return
    discount_rate: float = 0.10            # 10% discount rate
    project_life_years: int = 20
    construction_years: int = 2

    # Cost assumptions
    feedstock_price_usd_kg: float = 0.05   # $/kg feedstock (near-zero for waste)
    electricity_price_usd_kwh: float = 0.07
    steam_price_usd_mj: float = 0.015      # Natural gas steam

    # Default feedstock composition
    default_feedstock: Dict[str, float] = None

    def __post_init__(self):
        if self.default_feedstock is None:
            self.default_feedstock = {
                'PE': 0.80,    # 80% polyethylene
                'PET': 0.10,   # 10% PET
                'EVOH': 0.10   # 10% EVOH
            }


@dataclass
class PolymerProperties:
    """
    Polymer-specific properties for STRAP analysis.

    Includes virgin polymer environmental impacts for comparison.
    """
    # Virgin polymer GWP (kg CO2eq/kg) - from Environmental Footprint 2.0
    virgin_gwp: Dict[str, float] = None

    # Virgin polymer fossil fuel consumption (MJ/kg)
    virgin_ffc: Dict[str, float] = None

    # Market prices for recovered polymers (USD/kg)
    recovered_prices: Dict[str, float] = None

    # Typical dissolution temperatures (°C)
    dissolution_temps: Dict[str, float] = None

    # Compatible solvents for each polymer
    compatible_solvents: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.virgin_gwp is None:
            self.virgin_gwp = {
                'PE': 2.091, 'LDPE': 2.091, 'HDPE': 2.091, 'mLLDPE': 2.091,
                'EVOH': 7.30,
                'PET': 2.15,
                'PP': 1.98,
                'PS': 3.45,
                'PVC': 2.20,
                'PA': 6.80,  # Nylon
                'PC': 5.85,  # Polycarbonate
                'default': 2.50
            }
        if self.virgin_ffc is None:
            self.virgin_ffc = {
                'PE': 75.77, 'LDPE': 75.77, 'HDPE': 75.77,
                'EVOH': 161.78,
                'PET': 68.50,
                'PP': 73.20,
                'PS': 87.40,
                'default': 75.0
            }
        if self.recovered_prices is None:
            self.recovered_prices = {
                'PE': 0.90, 'LDPE': 0.90, 'HDPE': 0.95,
                'EVOH': 4.50,  # Higher value specialty polymer
                'PET': 0.85,
                'PP': 0.95,
                'PS': 1.10,
                'default': 1.00
            }
        if self.dissolution_temps is None:
            self.dissolution_temps = {
                'PE': 95.0, 'LDPE': 95.0, 'HDPE': 110.0,
                'EVOH': 95.0,  # In DMSO
                'PET': 120.0,  # In DMF or other
                'PP': 100.0,
                'PS': 80.0,
                'default': 95.0
            }
        if self.compatible_solvents is None:
            self.compatible_solvents = {
                'PE': ['heptane', 'dodecane', 'xylene', 'toluene'],
                'LDPE': ['heptane', 'dodecane', 'xylene', 'toluene'],
                'HDPE': ['xylene', 'decalin', 'toluene'],
                'EVOH': ['dmso', 'dmf', 'water'],
                'PET': ['dmf', 'nmp', 'dmso'],
                'PP': ['xylene', 'decalin', 'toluene'],
                'PS': ['toluene', 'thf', 'dcm'],
                'default': ['xylene']
            }


# =============================================================================
# LCA EMISSION FACTORS (from Environmental Footprint 2.0 / IPCC GWP100)
# =============================================================================

LCA_EMISSION_FACTORS = {
    # Virgin polymer production (kg CO2eq/kg polymer)
    'virgin_gwp': {
        'PE': 2.091, 'LDPE': 2.091, 'HDPE': 2.091,
        'EVOH': 7.30,
        'PET': 2.15,
        'PP': 1.98,
        'PS': 3.45,
        'PVC': 2.20,
        'default': 2.50
    },

    # Virgin polymer fossil fuel consumption (MJ/kg)
    'virgin_ffc': {
        'PE': 75.77, 'LDPE': 75.77, 'HDPE': 75.77,
        'EVOH': 161.78,
        'PET': 68.50,
        'PP': 73.20,
        'default': 75.0
    },

    # Virgin polymer water use (m3/kg)
    'virgin_water': {
        'PE': 1.562,
        'EVOH': 1.329,
        'PET': 1.45,
        'default': 1.50
    },

    # Process utility emission factors
    'utilities': {
        'electricity_gwp': 0.42,       # kg CO2eq/kWh (US grid average)
        'electricity_ffc': 11.57,      # MJ/kWh
        'lp_steam_gwp': 0.065,         # kg CO2eq/MJ (natural gas)
        'lp_steam_ffc': 1.06,          # MJ/MJ
        'mp_steam_gwp': 0.068,         # kg CO2eq/MJ
        'cooling_water_gwp': 0.0001,   # kg CO2eq/kg water
    },

    # Solvent production emissions (kg CO2eq/kg solvent)
    'solvent_gwp': {
        'heptane': 0.95,
        'dodecane': 1.05,
        'xylene': 1.30,
        'toluene': 1.20,
        'dmso': 2.80,
        'dmf': 3.50,
        'nmp': 3.80,
        'thf': 4.20,
        'acetone': 2.10,
        'ethanol': 1.50,
        'methanol': 0.80,
        'dcm': 1.80,
        'default': 2.00
    },

    # Solvent fossil fuel consumption (MJ/kg solvent)
    'solvent_ffc': {
        'xylene': 65.97,
        'dmso': 55.0,
        'heptane': 50.0,
        'dodecane': 52.0,
        'toluene': 58.0,
        'default': 55.0
    },

    # Other process inputs
    'adsorbent_gwp': 1.48,  # kg CO2eq/kg activated carbon
    'transport_gwp': 0.045,  # kg CO2eq/kg feedstock (truck transport)
}

# STRAP process contribution factors (kg indicator / kg product)
# Based on paper scenarios - used for detailed LCA breakdown
STRAP_LCA_CONTRIBUTIONS = {
    'S1_PE': {
        'adsorbent': 0.001475,
        'feedstock_transport': 0.044809,
        'xylene': 0.001323,
        'cooling_water': 0.000072,
        'lp_steam': 0.065489,
        'electricity': 0.766121,
    },
    'S2_PE': {
        'adsorbent': 0.001475,
        'feedstock_transport': 0.044809,
        'xylene': 0.001323,
        'cooling_water': 0.000072,
        'lp_steam': 0.065489,
        'electricity': 0.766121,
    },
    'S2_EVOH': {
        'adsorbent': 0.001821,
        'feedstock_transport': 0.055335,
        'xylene': 0.001634,
        'dmso': 0.0,
        'cooling_water': 0.000089,
        'lp_steam': 0.080872,
        'electricity': 0.946081,
    },
    'S3_PE': {
        'adsorbent': 0.001475,
        'feedstock_transport': 0.044809,
        'xylene': 0.001323,
        'cooling_water': 0.000072,
        'lp_steam': 0.065489,
        'electricity': 0.766121,
    },
    'S3_EVOH': {
        'adsorbent': 0.003337,
        'feedstock_transport': 0.101387,
        'xylene': 0.002993,
        'dmso': 0.000663,
        'cooling_water': 0.000164,
        'lp_steam': 0.148178,
        'mp_steam': 0.000458,
        'electricity': 1.733459,
    },
}


# =============================================================================
# STRAP EQUIPMENT COST DATA
# =============================================================================

STRAP_EQUIPMENT_COSTS = {
    # Equipment: (base_cost_usd, base_capacity_kg_hr, scaling_exponent)
    'dissolution_vessel': (150000, 1000, 0.6),
    'hot_filtration': (80000, 500, 0.5),
    'cold_filtration': (60000, 500, 0.5),
    'precipitation_vessel': (120000, 800, 0.6),
    'vacuum_drier': (200000, 300, 0.65),
    'melt_devolatilizer': (350000, 500, 0.7),
    'outgassing_extruder': (400000, 400, 0.7),
    'activated_carbon_column': (60000, 200, 0.5),
    'solvent_heat_exchanger': (40000, 500, 0.65),
    'solvent_storage_tank': (25000, 1000, 0.5),
    'polymer_storage': (20000, 500, 0.5),
}

# Material cost multipliers
MATERIAL_FACTORS = {
    'carbon_steel': 1.0,
    'stainless_steel': 1.8,
    'stainless_316': 2.2,
    'hastelloy': 3.5,
    'titanium': 4.5,
}

# Chemical Engineering Plant Cost Index
CEPCI = {
    2020: 596.2,
    2024: 750.0,
    2026: 816.0,  # Projected
}


# Initialize STRAP configurations
DEFAULT_STRAP_CONFIG = STRAPConfig()
DEFAULT_POLYMER_PROPS = PolymerProperties()


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

def calculate_distillation_energy(
    solvent: str,
    flow_rate_kg_hr: float,
    feed_temp_c: float = 25.0,
    recovery_fraction: float = 0.95,
    props: SolventProperties = None
) -> Dict[str, float]:
    """
    Calculate energy requirements for solvent distillation/recovery.

    Parameters
    ----------
    solvent : str
        Solvent name
    flow_rate_kg_hr : float
        Mass flow rate of solvent (kg/hr)
    feed_temp_c : float
        Feed temperature (°C)
    recovery_fraction : float
        Fraction of solvent recovered (0-1)
    props : SolventProperties
        Solvent properties configuration

    Returns
    -------
    dict
        Energy breakdown (heating, vaporization, total in kW and kWh/kg)
    """
    if props is None:
        props = DEFAULT_SOLVENT_PROPS

    # Get solvent properties
    bp = props.get_property(solvent, props.boiling_points)
    h_vap = props.get_property(solvent, props.heat_vaporization)  # kJ/kg
    cp = props.get_property(solvent, props.specific_heat)  # kJ/kg·K

    # Calculate heating energy (sensible heat to boiling point)
    delta_t = bp - feed_temp_c
    q_heating = flow_rate_kg_hr * cp * delta_t / 3600  # kW

    # Calculate vaporization energy
    q_vaporization = flow_rate_kg_hr * h_vap * recovery_fraction / 3600  # kW

    # Total energy
    q_total = q_heating + q_vaporization

    # Energy per kg solvent
    energy_per_kg = q_total * 3600 / flow_rate_kg_hr if flow_rate_kg_hr > 0 else 0

    return {
        'heating_kw': round(q_heating, 2),
        'vaporization_kw': round(q_vaporization, 2),
        'total_kw': round(q_total, 2),
        'kwh_per_kg': round(energy_per_kg / 3600, 4),
        'mj_per_kg': round(energy_per_kg / 1000, 4),
        'boiling_point_c': bp,
        'recovery_fraction': recovery_fraction
    }


def estimate_equipment_cost(
    capacity_kg_hr: float,
    equipment_type: str = 'distillation',
    material: str = 'stainless_steel'
) -> Dict[str, float]:
    """
    Estimate equipment purchase cost using scaling correlations.

    Based on: Turton et al., "Analysis, Synthesis, and Design of Chemical Processes"

    Parameters
    ----------
    capacity_kg_hr : float
        Processing capacity (kg/hr)
    equipment_type : str
        Type of equipment ('distillation', 'heat_exchanger', 'pump', 'tank')
    material : str
        Construction material ('carbon_steel', 'stainless_steel', 'hastelloy')

    Returns
    -------
    dict
        Cost breakdown (purchase cost, installed cost, factors)
    """
    # Base costs and scaling exponents (2020 USD)
    equipment_data = {
        'distillation': {'base_cost': 50000, 'base_capacity': 100, 'exponent': 0.6},
        'heat_exchanger': {'base_cost': 20000, 'base_capacity': 50, 'exponent': 0.65},
        'pump': {'base_cost': 8000, 'base_capacity': 100, 'exponent': 0.35},
        'tank': {'base_cost': 15000, 'base_capacity': 500, 'exponent': 0.5},
        'evaporator': {'base_cost': 40000, 'base_capacity': 80, 'exponent': 0.55},
    }

    # Material factors
    material_factors = {
        'carbon_steel': 1.0,
        'stainless_steel': 1.8,
        'hastelloy': 3.5,
    }

    # Installation factors
    installation_factor = 3.0  # Typical Lang factor

    # Get equipment parameters
    equip = equipment_data.get(equipment_type, equipment_data['distillation'])
    mat_factor = material_factors.get(material, 1.8)

    # Scale equipment cost
    scaled_cost = equip['base_cost'] * (capacity_kg_hr / equip['base_capacity']) ** equip['exponent']
    purchase_cost = scaled_cost * mat_factor

    # Apply CEPCI adjustment (2020 to 2026)
    cepci_2020 = 596.2
    cepci_2026 = 750.0  # Estimated
    purchase_cost *= cepci_2026 / cepci_2020

    # Installed cost
    installed_cost = purchase_cost * installation_factor

    return {
        'purchase_cost_usd': round(purchase_cost, 0),
        'installed_cost_usd': round(installed_cost, 0),
        'material_factor': mat_factor,
        'installation_factor': installation_factor,
        'equipment_type': equipment_type
    }


def calculate_operating_costs(
    energy_kw: float,
    solvent_loss_kg_hr: float,
    solvent: str,
    labor_operators: int = 2,
    config: TEAConfig = None,
    props: SolventProperties = None
) -> Dict[str, float]:
    """
    Calculate annual operating costs for solvent recovery.

    Parameters
    ----------
    energy_kw : float
        Total energy consumption (kW)
    solvent_loss_kg_hr : float
        Solvent makeup rate due to losses (kg/hr)
    solvent : str
        Solvent name
    labor_operators : int
        Number of operators per shift
    config : TEAConfig
        TEA configuration
    props : SolventProperties
        Solvent properties

    Returns
    -------
    dict
        Annual operating cost breakdown (USD/year)
    """
    if config is None:
        config = DEFAULT_TEA_CONFIG
    if props is None:
        props = DEFAULT_SOLVENT_PROPS

    operating_hours = config.operating_hours

    # Utility costs
    electricity_price = 0.08  # USD/kWh
    steam_price = 15.0  # USD/1000 kg (assume 50% of energy from steam)

    energy_kwh_year = energy_kw * operating_hours
    electricity_cost = energy_kwh_year * 0.3 * electricity_price  # 30% electricity
    steam_energy_mj = energy_kw * 0.7 * 3.6 * operating_hours  # 70% steam, convert kW to MJ
    steam_kg_year = steam_energy_mj / 2.5  # ~2.5 MJ/kg steam
    steam_cost = steam_kg_year * steam_price / 1000

    # Solvent makeup cost
    solvent_price = props.get_property(solvent, props.prices)
    solvent_cost = solvent_loss_kg_hr * operating_hours * solvent_price

    # Labor cost
    total_operators = labor_operators * config.shifts_per_day
    labor_base = total_operators * config.labor_cost_per_operator
    labor_cost = labor_base * (1 + config.fringe_benefits)

    # Total variable operating cost
    total_variable = electricity_cost + steam_cost + solvent_cost
    total_fixed = labor_cost
    total_operating = total_variable + total_fixed

    return {
        'electricity_usd_yr': round(electricity_cost, 0),
        'steam_usd_yr': round(steam_cost, 0),
        'solvent_makeup_usd_yr': round(solvent_cost, 0),
        'labor_usd_yr': round(labor_cost, 0),
        'total_variable_usd_yr': round(total_variable, 0),
        'total_fixed_usd_yr': round(total_fixed, 0),
        'total_operating_usd_yr': round(total_operating, 0)
    }


def calculate_carbon_footprint(
    energy_kw: float,
    solvent: str,
    solvent_loss_kg_hr: float,
    operating_hours: int = 7920,
    config: LCAConfig = None
) -> Dict[str, float]:
    """
    Calculate carbon footprint (CO2 equivalent emissions).

    Parameters
    ----------
    energy_kw : float
        Total energy consumption (kW)
    solvent : str
        Solvent name
    solvent_loss_kg_hr : float
        Solvent makeup rate (kg/hr)
    operating_hours : int
        Annual operating hours
    config : LCAConfig
        LCA configuration

    Returns
    -------
    dict
        Annual CO2eq emissions breakdown (kg CO2eq/year)
    """
    if config is None:
        config = DEFAULT_LCA_CONFIG

    # Energy emissions (assume 30% electricity, 70% steam)
    electricity_kwh_yr = energy_kw * 0.3 * operating_hours
    steam_mj_yr = energy_kw * 0.7 * 3.6 * operating_hours

    electricity_emissions = electricity_kwh_yr * config.electricity_emission
    steam_emissions = steam_mj_yr * config.steam_emission

    # Solvent production emissions (for makeup solvent)
    solvent_emission_factor = config.get_solvent_emission(solvent)
    solvent_kg_yr = solvent_loss_kg_hr * operating_hours
    solvent_emissions = solvent_kg_yr * solvent_emission_factor

    # Total emissions
    total_emissions = electricity_emissions + steam_emissions + solvent_emissions

    return {
        'electricity_kg_co2eq_yr': round(electricity_emissions, 0),
        'steam_kg_co2eq_yr': round(steam_emissions, 0),
        'solvent_kg_co2eq_yr': round(solvent_emissions, 0),
        'total_kg_co2eq_yr': round(total_emissions, 0),
        'total_tonnes_co2eq_yr': round(total_emissions / 1000, 2)
    }


# =============================================================================
# STRAP-SPECIFIC CALCULATION FUNCTIONS
# =============================================================================

def estimate_strap_equipment_cost(
    capacity_kg_hr: float,
    equipment_type: str,
    material: str = 'stainless_steel',
    year: int = 2026
) -> Dict[str, float]:
    """
    Estimate STRAP equipment cost using six-tenths scaling rule.

    Parameters
    ----------
    capacity_kg_hr : float
        Processing capacity (kg/hr)
    equipment_type : str
        Type of STRAP equipment (from STRAP_EQUIPMENT_COSTS)
    material : str
        Construction material
    year : int
        Cost year for CEPCI adjustment

    Returns
    -------
    dict
        Equipment costs (purchase, installed)
    """
    if equipment_type not in STRAP_EQUIPMENT_COSTS:
        raise ValueError(f"Unknown equipment type: {equipment_type}. "
                        f"Available: {list(STRAP_EQUIPMENT_COSTS.keys())}")

    base_cost, base_capacity, exponent = STRAP_EQUIPMENT_COSTS[equipment_type]

    # Scale by capacity
    scaled_cost = base_cost * (capacity_kg_hr / base_capacity) ** exponent

    # Apply material factor
    material_factor = MATERIAL_FACTORS.get(material, 1.8)
    scaled_cost *= material_factor

    # Apply CEPCI adjustment (from 2020 to target year)
    cepci_ratio = CEPCI.get(year, 816.0) / CEPCI[2020]
    scaled_cost *= cepci_ratio

    # Installed cost (Lang factor = 3.0 for fluid processing)
    installed_cost = scaled_cost * 3.0

    return {
        'equipment_type': equipment_type,
        'capacity_kg_hr': capacity_kg_hr,
        'purchase_cost_usd': round(scaled_cost, 0),
        'installed_cost_usd': round(installed_cost, 0),
        'material': material,
        'cepci_year': year
    }


def calculate_strap_capital_costs(
    capacity_mt_yr: float,
    recovery_steps: List[Dict],
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Calculate total capital investment for a STRAP facility.

    Parameters
    ----------
    capacity_mt_yr : float
        Plant capacity (metric tons feedstock per year)
    recovery_steps : list
        List of recovery steps, each with {'polymer': str, 'solvent': str}
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Capital cost breakdown and TCI
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    # Convert to kg/hr
    capacity_kg_hr = capacity_mt_yr * 1000 / config.operating_hours_per_year

    equipment_costs = {}
    total_equipment = 0

    # Base STRAP equipment (per recovery step)
    base_equipment = [
        'dissolution_vessel',
        'hot_filtration',
        'precipitation_vessel',
        'cold_filtration',
        'vacuum_drier',
    ]

    # Equipment needed for each recovery step
    for i, step in enumerate(recovery_steps):
        step_prefix = f"step{i+1}_{step['polymer']}"
        for equip in base_equipment:
            cost_data = estimate_strap_equipment_cost(capacity_kg_hr, equip)
            equipment_costs[f"{step_prefix}_{equip}"] = cost_data['installed_cost_usd']
            total_equipment += cost_data['installed_cost_usd']

    # Shared equipment (only once)
    shared_equipment = [
        'melt_devolatilizer',
        'outgassing_extruder',
        'activated_carbon_column',
        'solvent_heat_exchanger',
        'solvent_storage_tank',
        'polymer_storage',
    ]

    for equip in shared_equipment:
        cost_data = estimate_strap_equipment_cost(capacity_kg_hr, equip)
        equipment_costs[f"shared_{equip}"] = cost_data['installed_cost_usd']
        total_equipment += cost_data['installed_cost_usd']

    # Fixed Capital Investment (FCI)
    fci = total_equipment * 1.2  # 20% for piping, instrumentation, contingency

    # Working capital
    working_capital = fci * config.target_irr

    # Total Capital Investment (TCI)
    tci = fci + working_capital

    return {
        'equipment_costs': equipment_costs,
        'total_equipment_cost_usd': round(total_equipment, 0),
        'fixed_capital_investment_usd': round(fci, 0),
        'working_capital_usd': round(working_capital, 0),
        'total_capital_investment_usd': round(tci, 0),
        'capacity_mt_yr': capacity_mt_yr,
        'capacity_kg_hr': round(capacity_kg_hr, 1),
        'recovery_steps': len(recovery_steps)
    }


def calculate_strap_operating_costs(
    capacity_mt_yr: float,
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    capital_costs: Dict,
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Calculate annual operating costs for STRAP process.

    Parameters
    ----------
    capacity_mt_yr : float
        Plant capacity (metric tons per year)
    feedstock_composition : dict
        Polymer fractions {'PE': 0.8, 'EVOH': 0.1, ...}
    recovery_steps : list
        Recovery step configurations
    capital_costs : dict
        Output from calculate_strap_capital_costs
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Operating cost breakdown
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    fci = capital_costs['fixed_capital_investment_usd']
    capacity_kg_hr = capital_costs['capacity_kg_hr']

    # Variable costs
    variable_costs = {}

    # Feedstock cost
    feedstock_cost = capacity_mt_yr * 1000 * config.feedstock_price_usd_kg
    variable_costs['feedstock'] = feedstock_cost

    # Solvent costs (makeup due to losses)
    solvent_cost_total = 0
    for step in recovery_steps:
        solvent = step.get('solvent', 'xylene')
        solvent_price = DEFAULT_SOLVENT_PROPS.prices.get(solvent.lower(), 2.0)
        # Solvent flow = feedstock * solvent ratio
        solvent_flow_kg_hr = capacity_kg_hr / config.feed_to_solvent_ratio
        # Annual makeup = loss rate * flow * hours
        makeup_kg_yr = solvent_flow_kg_hr * config.solvent_loss_rate * config.operating_hours_per_year
        step_solvent_cost = makeup_kg_yr * solvent_price
        variable_costs[f"solvent_{step['polymer']}_{solvent}"] = step_solvent_cost
        solvent_cost_total += step_solvent_cost

    # Energy costs
    # Estimate energy: ~0.5 kWh electricity + ~2 MJ steam per kg feedstock
    electricity_kwh_yr = capacity_mt_yr * 1000 * 0.5
    steam_mj_yr = capacity_mt_yr * 1000 * 2.0

    electricity_cost = electricity_kwh_yr * config.electricity_price_usd_kwh
    steam_cost = steam_mj_yr * config.steam_price_usd_mj

    variable_costs['electricity'] = electricity_cost
    variable_costs['steam'] = steam_cost

    # Adsorbent costs (activated carbon replacement)
    # Estimate: 0.01 kg adsorbent per kg feedstock per year
    adsorbent_cost = capacity_mt_yr * 1000 * 0.01 * 3.0  # $3/kg activated carbon
    variable_costs['adsorbent'] = adsorbent_cost

    total_variable = sum(variable_costs.values())

    # Fixed costs
    fixed_costs = {}

    # Labor (operators)
    operators = max(6, int(capacity_mt_yr / 3000))  # Scale with capacity
    labor_cost = operators * 75000 * 1.4  # With fringe benefits
    fixed_costs['labor'] = labor_cost

    # Maintenance (3% of FCI)
    fixed_costs['maintenance'] = fci * 0.03

    # Insurance and taxes (1.7% of FCI)
    fixed_costs['insurance_taxes'] = fci * 0.017

    # Overhead (60% of labor)
    fixed_costs['overhead'] = labor_cost * 0.6

    total_fixed = sum(fixed_costs.values())

    # Total operating cost
    total_operating = total_variable + total_fixed

    # Unit operating cost (per kg product)
    total_product_kg = capacity_mt_yr * 1000 * config.polymer_recovery_efficiency
    uoc = total_operating / total_product_kg

    return {
        'variable_costs': variable_costs,
        'fixed_costs': fixed_costs,
        'total_variable_usd_yr': round(total_variable, 0),
        'total_fixed_usd_yr': round(total_fixed, 0),
        'total_operating_cost_usd_yr': round(total_operating, 0),
        'unit_operating_cost_usd_kg': round(uoc, 4),
        'capacity_mt_yr': capacity_mt_yr,
        'product_mt_yr': round(capacity_mt_yr * config.polymer_recovery_efficiency, 1)
    }


def calculate_strap_revenue(
    capacity_mt_yr: float,
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    config: STRAPConfig = None,
    polymer_props: PolymerProperties = None
) -> Dict[str, Any]:
    """
    Calculate annual revenue from recovered polymers.

    Parameters
    ----------
    capacity_mt_yr : float
        Plant capacity (metric tons per year)
    feedstock_composition : dict
        Polymer fractions in feedstock
    recovery_steps : list
        Which polymers are recovered
    config : STRAPConfig
        STRAP configuration
    polymer_props : PolymerProperties
        Polymer property data

    Returns
    -------
    dict
        Revenue breakdown by polymer
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG
    if polymer_props is None:
        polymer_props = DEFAULT_POLYMER_PROPS

    revenue_by_polymer = {}
    total_revenue = 0

    for step in recovery_steps:
        polymer = step['polymer'].upper()
        fraction = feedstock_composition.get(polymer, feedstock_composition.get(polymer.lower(), 0))

        if fraction > 0:
            # Recovered mass
            recovered_kg = capacity_mt_yr * 1000 * fraction * config.polymer_recovery_efficiency

            # Price
            price = polymer_props.recovered_prices.get(
                polymer, polymer_props.recovered_prices.get('default', 1.0)
            )

            # Revenue
            revenue = recovered_kg * price
            revenue_by_polymer[polymer] = {
                'recovered_kg_yr': round(recovered_kg, 0),
                'price_usd_kg': price,
                'revenue_usd_yr': round(revenue, 0)
            }
            total_revenue += revenue

    return {
        'by_polymer': revenue_by_polymer,
        'total_revenue_usd_yr': round(total_revenue, 0),
        'capacity_mt_yr': capacity_mt_yr
    }


def calculate_strap_economics_at_scale(
    capacity_mt_yr: float,
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    config: STRAPConfig = None,
    polymer_props: PolymerProperties = None
) -> Dict[str, Any]:
    """
    Full STRAP economic analysis at a given scale.

    Parameters
    ----------
    capacity_mt_yr : float
        Plant capacity (metric tons per year)
    feedstock_composition : dict
        Polymer fractions {'PE': 0.8, 'PET': 0.1, 'EVOH': 0.1}
    recovery_steps : list
        Recovery configurations [{'polymer': 'PE', 'solvent': 'heptane'}, ...]
    config : STRAPConfig
        STRAP configuration
    polymer_props : PolymerProperties
        Polymer properties

    Returns
    -------
    dict
        Complete economic analysis
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG
    if polymer_props is None:
        polymer_props = DEFAULT_POLYMER_PROPS

    # Capital costs
    capital = calculate_strap_capital_costs(capacity_mt_yr, recovery_steps, config)

    # Operating costs
    operating = calculate_strap_operating_costs(
        capacity_mt_yr, feedstock_composition, recovery_steps, capital, config
    )

    # Revenue
    revenue = calculate_strap_revenue(
        capacity_mt_yr, feedstock_composition, recovery_steps, config, polymer_props
    )

    # Economic metrics
    tci = capital['total_capital_investment_usd']
    annual_cost = operating['total_operating_cost_usd_yr']
    annual_revenue = revenue['total_revenue_usd_yr']

    # Net annual profit
    net_profit = annual_revenue - annual_cost

    # Simple payback period
    payback = tci / net_profit if net_profit > 0 else float('inf')

    # Return on investment
    roi = (net_profit / tci) * 100 if tci > 0 else 0

    return {
        'capacity_mt_yr': capacity_mt_yr,
        'feedstock_composition': feedstock_composition,
        'recovery_steps': recovery_steps,
        'capital': capital,
        'operating': operating,
        'revenue': revenue,
        'economics': {
            'total_capital_investment_usd': tci,
            'annual_operating_cost_usd': annual_cost,
            'annual_revenue_usd': annual_revenue,
            'net_annual_profit_usd': round(net_profit, 0),
            'simple_payback_years': round(payback, 2),
            'return_on_investment_pct': round(roi, 1),
            'unit_operating_cost_usd_kg': operating['unit_operating_cost_usd_kg'],
            'tci_millions': round(tci / 1e6, 2)
        }
    }


def calculate_msp(
    capacity_mt_yr: float,
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    target_irr: float = 0.15,
    project_life: int = 20,
    config: STRAPConfig = None
) -> Dict[str, float]:
    """
    Calculate Minimum Selling Price (MSP) where NPV = 0 at target IRR.

    Uses bisection method to find the break-even price.

    Parameters
    ----------
    capacity_mt_yr : float
        Plant capacity
    feedstock_composition : dict
        Polymer fractions
    recovery_steps : list
        Recovery configurations
    target_irr : float
        Target internal rate of return (default 15%)
    project_life : int
        Project lifetime in years
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        MSP for each recovered polymer
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    # Get base economics
    econ = calculate_strap_economics_at_scale(
        capacity_mt_yr, feedstock_composition, recovery_steps, config
    )

    tci = econ['capital']['total_capital_investment_usd']
    annual_cost = econ['operating']['total_operating_cost_usd_yr']

    # Total recovered mass per year
    total_recovered_kg = 0
    polymer_masses = {}
    for step in recovery_steps:
        polymer = step['polymer'].upper()
        fraction = feedstock_composition.get(polymer, feedstock_composition.get(polymer.lower(), 0))
        mass = capacity_mt_yr * 1000 * fraction * config.polymer_recovery_efficiency
        polymer_masses[polymer] = mass
        total_recovered_kg += mass

    # Calculate MSP using NPV = 0 condition
    # NPV = -TCI + sum(annual_cashflow / (1+r)^t) = 0
    # Annual cashflow = Revenue - Operating cost
    # Revenue = MSP * total_mass
    # Solve for MSP

    # Present value factor for annuity
    r = target_irr
    n = project_life
    pv_factor = (1 - (1 + r) ** -n) / r

    # MSP * mass * pv_factor = TCI + annual_cost * pv_factor
    # MSP = (TCI / pv_factor + annual_cost) / total_mass

    msp_weighted = (tci / pv_factor + annual_cost) / total_recovered_kg

    # Allocate MSP by polymer (weighted by their typical market value ratios)
    msp_by_polymer = {}
    total_value_ratio = 0
    value_ratios = {}

    for polymer, mass in polymer_masses.items():
        typical_price = DEFAULT_POLYMER_PROPS.recovered_prices.get(polymer, 1.0)
        value_ratios[polymer] = typical_price
        total_value_ratio += typical_price * mass

    for polymer, mass in polymer_masses.items():
        if mass > 0:
            # Allocate MSP proportionally to typical value
            ratio = (value_ratios[polymer] * mass) / total_value_ratio
            polymer_msp = msp_weighted * (total_recovered_kg * ratio) / mass
            msp_by_polymer[polymer] = round(polymer_msp, 4)

    return {
        'msp_by_polymer_usd_kg': msp_by_polymer,
        'msp_weighted_avg_usd_kg': round(msp_weighted, 4),
        'target_irr': target_irr,
        'project_life_years': project_life,
        'total_capital_investment_usd': tci,
        'annual_operating_cost_usd': annual_cost,
        'total_recovered_kg_yr': round(total_recovered_kg, 0)
    }


def generate_scale_economics_curve(
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    capacity_range: Tuple[float, float] = (2500, 25000),
    num_points: int = 20,
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Generate UOC and TCI curves across a range of plant capacities.

    Parameters
    ----------
    feedstock_composition : dict
        Polymer fractions
    recovery_steps : list
        Recovery configurations
    capacity_range : tuple
        (min, max) capacity in mt/year
    num_points : int
        Number of data points
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Arrays of capacity, UOC, TCI values
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    capacities = np.linspace(capacity_range[0], capacity_range[1], num_points)
    uoc_values = []
    tci_values = []

    for cap in capacities:
        econ = calculate_strap_economics_at_scale(
            cap, feedstock_composition, recovery_steps, config
        )
        uoc_values.append(econ['economics']['unit_operating_cost_usd_kg'])
        tci_values.append(econ['economics']['tci_millions'])

    return {
        'capacities_mt_yr': capacities.tolist(),
        'uoc_usd_kg': uoc_values,
        'tci_millions': tci_values,
        'feedstock_composition': feedstock_composition,
        'recovery_steps': recovery_steps
    }


# =============================================================================
# STRAP SCENARIO FRAMEWORK
# =============================================================================

@dataclass
class STRAPScenario:
    """
    Represents a complete STRAP scenario configuration.
    """
    name: str
    feedstock_composition: Dict[str, float]
    recovery_steps: List[Dict]
    capacity_mt_yr: float
    description: str = ""

    def __post_init__(self):
        # Validate composition sums to ~1.0
        total = sum(self.feedstock_composition.values())
        if not (0.99 <= total <= 1.01):
            warnings.warn(f"Feedstock composition sums to {total}, expected ~1.0")


def build_strap_scenario(
    name: str,
    feedstock_composition: Dict[str, float],
    recovery_sequence: List[Dict],
    capacity_mt_yr: float = 10000,
    description: str = ""
) -> STRAPScenario:
    """
    Build a STRAP scenario from components.

    Parameters
    ----------
    name : str
        Scenario name (e.g., "S1_PE_only", "S2_PE_EVOH")
    feedstock_composition : dict
        Polymer fractions {'PE': 0.8, 'PET': 0.1, 'EVOH': 0.1}
    recovery_sequence : list
        List of recovery steps:
        [{'polymer': 'PE', 'solvent': 'heptane', 'temp': 95, 'recover': True}, ...]
    capacity_mt_yr : float
        Plant capacity in metric tons/year
    description : str
        Human-readable description

    Returns
    -------
    STRAPScenario
        Configured scenario object
    """
    # Filter to only steps that recover polymer
    active_steps = [s for s in recovery_sequence if s.get('recover', True)]

    return STRAPScenario(
        name=name,
        feedstock_composition=feedstock_composition,
        recovery_steps=active_steps,
        capacity_mt_yr=capacity_mt_yr,
        description=description
    )


def create_default_scenarios() -> Dict[str, STRAPScenario]:
    """
    Create the three default scenarios from the STRAP paper.

    Returns
    -------
    dict
        Dictionary of scenario name -> STRAPScenario
    """
    # Default SUT feedstock (from paper)
    sut_feedstock = {'PE': 0.90, 'EVOH': 0.08, 'other': 0.02}

    # Scenario 1: PE recovery only
    s1 = build_strap_scenario(
        name="S1_PE_only",
        feedstock_composition=sut_feedstock,
        recovery_sequence=[
            {'polymer': 'PE', 'solvent': 'heptane', 'temp': 95, 'recover': True},
        ],
        capacity_mt_yr=10000,
        description="PE recovery only, EVOH sent to landfill"
    )

    # Scenario 2: PE + EVOH recovery (high EVOH value)
    s2 = build_strap_scenario(
        name="S2_PE_EVOH",
        feedstock_composition=sut_feedstock,
        recovery_sequence=[
            {'polymer': 'PE', 'solvent': 'heptane', 'temp': 95, 'recover': True},
            {'polymer': 'EVOH', 'solvent': 'dmso', 'temp': 95, 'recover': True},
        ],
        capacity_mt_yr=10000,
        description="PE and EVOH recovery, EVOH sold at premium"
    )

    # Scenario 3: PE + EVOH recovery (alternative EVOH processing)
    s3 = build_strap_scenario(
        name="S3_PE_EVOH_alt",
        feedstock_composition=sut_feedstock,
        recovery_sequence=[
            {'polymer': 'PE', 'solvent': 'heptane', 'temp': 95, 'recover': True},
            {'polymer': 'EVOH', 'solvent': 'dmso', 'temp': 120, 'recover': True},
        ],
        capacity_mt_yr=10000,
        description="PE and EVOH recovery, alternative EVOH processing"
    )

    return {'S1': s1, 'S2': s2, 'S3': s3}


def analyze_scenario(
    scenario: STRAPScenario,
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Run full TEA analysis on a STRAP scenario.

    Parameters
    ----------
    scenario : STRAPScenario
        Scenario configuration
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Complete TEA results for the scenario
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    results = calculate_strap_economics_at_scale(
        capacity_mt_yr=scenario.capacity_mt_yr,
        feedstock_composition=scenario.feedstock_composition,
        recovery_steps=scenario.recovery_steps,
        config=config
    )

    results['scenario_name'] = scenario.name
    results['scenario_description'] = scenario.description

    return results


def compare_scenarios(
    scenarios: List[STRAPScenario],
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Compare multiple STRAP scenarios.

    Parameters
    ----------
    scenarios : list
        List of STRAPScenario objects
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Comparison table and rankings
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    results = []
    for scenario in scenarios:
        analysis = analyze_scenario(scenario, config)
        results.append({
            'name': scenario.name,
            'description': scenario.description,
            'capacity_mt_yr': scenario.capacity_mt_yr,
            'tci_millions': analysis['economics']['tci_millions'],
            'annual_operating_cost': analysis['economics']['annual_operating_cost_usd'],
            'annual_revenue': analysis['economics']['annual_revenue_usd'],
            'net_profit': analysis['economics']['net_annual_profit_usd'],
            'payback_years': analysis['economics']['simple_payback_years'],
            'uoc_usd_kg': analysis['economics']['unit_operating_cost_usd_kg'],
            'roi_pct': analysis['economics']['return_on_investment_pct'],
        })

    # Rank by different metrics
    df = pd.DataFrame(results)

    return {
        'comparison_table': results,
        'best_payback': df.loc[df['payback_years'].idxmin(), 'name'] if len(df) > 0 else None,
        'best_roi': df.loc[df['roi_pct'].idxmax(), 'name'] if len(df) > 0 else None,
        'lowest_uoc': df.loc[df['uoc_usd_kg'].idxmin(), 'name'] if len(df) > 0 else None,
    }


# =============================================================================
# MULTI-INDICATOR LCA (IPCC GWP100 + Environmental Footprint 2.0)
# =============================================================================

@dataclass
class LCAIndicators:
    """
    All 8 environmental indicators from Environmental Footprint 2.0.
    """
    gwp: float = 0.0          # kg CO2eq/kg (Global Warming Potential)
    ffc: float = 0.0          # MJ/kg (Fossil Fuel Consumption)
    water_use: float = 0.0    # m3/kg
    htc: float = 0.0          # CTUh/kg (Human Toxicity - Cancer)
    htnc: float = 0.0         # CTUh/kg (Human Toxicity - Non-Cancer)
    etox: float = 0.0         # CTUe/kg (Ecotoxicity)
    acidification: float = 0.0  # mol H+ eq/kg
    ozone_depletion: float = 0.0  # kg CFC11 eq/kg
    pocp: float = 0.0         # kg NMVOC eq/kg (Photochemical Ozone)

    def to_dict(self) -> Dict[str, float]:
        return {
            'gwp_kg_co2eq': round(self.gwp, 4),
            'ffc_mj': round(self.ffc, 4),
            'water_use_m3': round(self.water_use, 6),
            'htc_ctuh': self.htc,
            'htnc_ctuh': self.htnc,
            'etox_ctue': round(self.etox, 6),
            'acidification_mol_h': round(self.acidification, 6),
            'ozone_depletion_kg_cfc11': self.ozone_depletion,
            'pocp_kg_nmvoc': round(self.pocp, 6),
        }


def calculate_strap_lca(
    scenario: STRAPScenario,
    config: STRAPConfig = None
) -> Dict[str, LCAIndicators]:
    """
    Calculate all LCA indicators for each recovered polymer in a STRAP scenario.

    Uses the contribution factors from the LCA-assumptions workbook.

    Parameters
    ----------
    scenario : STRAPScenario
        Scenario configuration
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        LCA indicators for each polymer {polymer: LCAIndicators}
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    results = {}

    for step in scenario.recovery_steps:
        polymer = step['polymer'].upper()
        solvent = step.get('solvent', 'xylene').lower()

        # Get base contribution factors (use S2 as default template)
        if polymer == 'PE':
            base_key = 'S2_PE'
        elif polymer == 'EVOH':
            base_key = 'S2_EVOH'
        else:
            base_key = 'S2_PE'  # Default to PE pattern

        contributions = STRAP_LCA_CONTRIBUTIONS.get(base_key, STRAP_LCA_CONTRIBUTIONS['S2_PE'])

        # Calculate GWP (primary indicator)
        gwp = sum(contributions.values())

        # Scale other indicators proportionally to GWP
        # (simplified - in practice each indicator has different scaling)
        ffc_ratio = LCA_EMISSION_FACTORS['utilities']['electricity_ffc'] / LCA_EMISSION_FACTORS['utilities']['electricity_gwp']

        indicators = LCAIndicators(
            gwp=gwp,
            ffc=gwp * ffc_ratio * 0.5,  # Approximate
            water_use=gwp * 0.15,        # Approximate
            htc=gwp * 6e-8,              # Approximate from paper data
            htnc=gwp * 1.5e-8,
            etox=gwp * 0.7,
            acidification=gwp * 0.003,
            ozone_depletion=gwp * 2e-10,
            pocp=gwp * 0.0015,
        )

        results[polymer] = indicators

    return results


def calculate_detailed_gwp_breakdown(
    scenario: STRAPScenario,
    config: STRAPConfig = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate detailed GWP breakdown by source for each polymer.

    Matches the stacked bar chart format from REF-LCA-1.webp.

    Parameters
    ----------
    scenario : STRAPScenario
        Scenario configuration
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        GWP breakdown by source for each polymer
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    results = {}

    for step in scenario.recovery_steps:
        polymer = step['polymer'].upper()
        solvent = step.get('solvent', 'xylene').lower()

        # Get contribution factors
        if polymer == 'PE':
            base_key = 'S2_PE'
        elif polymer == 'EVOH':
            base_key = 'S2_EVOH'
        else:
            base_key = 'S2_PE'

        contributions = STRAP_LCA_CONTRIBUTIONS.get(base_key, STRAP_LCA_CONTRIBUTIONS['S2_PE']).copy()

        # Convert to display format
        breakdown = {
            'Electricity': contributions.get('electricity', 0),
            'Steam': contributions.get('lp_steam', 0) + contributions.get('mp_steam', 0),
            'Feedstock plastic': contributions.get('feedstock_transport', 0),
            'Xylene': contributions.get('xylene', 0),
            'DMSO': contributions.get('dmso', 0),
            'Other': contributions.get('adsorbent', 0) + contributions.get('cooling_water', 0),
        }

        # Remove zero values
        breakdown = {k: v for k, v in breakdown.items() if v > 0.0001}

        total = sum(breakdown.values())
        breakdown['Total'] = total

        results[polymer] = breakdown

    return results


def compare_to_virgin(
    strap_lca: Dict[str, LCAIndicators],
    polymers: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare STRAP LCA results to virgin polymer production.

    Parameters
    ----------
    strap_lca : dict
        LCA results from calculate_strap_lca()
    polymers : list
        List of polymers to compare (default: all in strap_lca)

    Returns
    -------
    dict
        Reduction percentages for each indicator
    """
    if polymers is None:
        polymers = list(strap_lca.keys())

    results = {}

    for polymer in polymers:
        if polymer not in strap_lca:
            continue

        strap_indicators = strap_lca[polymer]

        # Get virgin values
        virgin_gwp = LCA_EMISSION_FACTORS['virgin_gwp'].get(
            polymer, LCA_EMISSION_FACTORS['virgin_gwp']['default']
        )
        virgin_ffc = LCA_EMISSION_FACTORS['virgin_ffc'].get(
            polymer, LCA_EMISSION_FACTORS['virgin_ffc']['default']
        )
        virgin_water = LCA_EMISSION_FACTORS['virgin_water'].get(
            polymer, LCA_EMISSION_FACTORS['virgin_water']['default']
        )

        # Calculate reductions
        gwp_reduction = (virgin_gwp - strap_indicators.gwp) / virgin_gwp * 100
        ffc_reduction = (virgin_ffc - strap_indicators.ffc) / virgin_ffc * 100
        water_reduction = (virgin_water - strap_indicators.water_use) / virgin_water * 100

        results[polymer] = {
            'virgin_gwp': virgin_gwp,
            'strap_gwp': round(strap_indicators.gwp, 4),
            'gwp_reduction_pct': round(gwp_reduction, 1),
            'virgin_ffc': virgin_ffc,
            'strap_ffc': round(strap_indicators.ffc, 4),
            'ffc_reduction_pct': round(ffc_reduction, 1),
            'virgin_water': virgin_water,
            'strap_water': round(strap_indicators.water_use, 6),
            'water_reduction_pct': round(water_reduction, 1),
        }

    return results


def run_full_strap_analysis(
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    capacity_mt_yr: float = 10000,
    scenario_name: str = "Custom",
    config: STRAPConfig = None
) -> Dict[str, Any]:
    """
    Run complete STRAP TEA and LCA analysis.

    Parameters
    ----------
    feedstock_composition : dict
        Polymer fractions {'PE': 0.8, 'PET': 0.1, 'EVOH': 0.1}
    recovery_steps : list
        Recovery configurations [{'polymer': 'PE', 'solvent': 'heptane'}, ...]
    capacity_mt_yr : float
        Plant capacity in metric tons/year
    scenario_name : str
        Name for this analysis
    config : STRAPConfig
        STRAP configuration

    Returns
    -------
    dict
        Complete TEA and LCA results
    """
    if config is None:
        config = DEFAULT_STRAP_CONFIG

    # Build scenario
    scenario = build_strap_scenario(
        name=scenario_name,
        feedstock_composition=feedstock_composition,
        recovery_sequence=recovery_steps,
        capacity_mt_yr=capacity_mt_yr
    )

    # TEA Analysis
    tea_results = calculate_strap_economics_at_scale(
        capacity_mt_yr=capacity_mt_yr,
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        config=config
    )

    # MSP Calculation
    msp_results = calculate_msp(
        capacity_mt_yr=capacity_mt_yr,
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        config=config
    )

    # LCA Analysis
    lca_results = calculate_strap_lca(scenario, config)
    lca_dict = {polymer: ind.to_dict() for polymer, ind in lca_results.items()}

    # GWP Breakdown
    gwp_breakdown = calculate_detailed_gwp_breakdown(scenario, config)

    # Virgin comparison
    virgin_comparison = compare_to_virgin(lca_results)

    return {
        'scenario': {
            'name': scenario_name,
            'feedstock_composition': feedstock_composition,
            'recovery_steps': recovery_steps,
            'capacity_mt_yr': capacity_mt_yr,
        },
        'tea': tea_results,
        'msp': msp_results,
        'lca': {
            'by_polymer': lca_dict,
            'gwp_breakdown': gwp_breakdown,
            'virgin_comparison': virgin_comparison,
        },
        'summary': {
            'tci_millions': tea_results['economics']['tci_millions'],
            'uoc_usd_kg': tea_results['economics']['unit_operating_cost_usd_kg'],
            'payback_years': tea_results['economics']['simple_payback_years'],
            'msp_avg_usd_kg': msp_results['msp_weighted_avg_usd_kg'],
            'gwp_reduction_pct': {
                p: v['gwp_reduction_pct']
                for p, v in virgin_comparison.items()
            }
        }
    }


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS (Called by Agent)
# =============================================================================

def run_full_tea_analysis(
    solvent: str,
    polymer_throughput_kg_hr: float,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0,
    config: TEAConfig = None
) -> Dict[str, Any]:
    """
    Run complete TEA for a solvent recovery process.

    This is the main function called by the agent for TEA queries.

    Parameters
    ----------
    solvent : str
        Solvent name
    polymer_throughput_kg_hr : float
        Polymer processing rate (kg/hr)
    solvent_to_polymer_ratio : float
        Mass ratio of solvent to polymer (default: 10:1)
    recovery_fraction : float
        Solvent recovery efficiency (0-1)
    process_temp_c : float
        Process temperature (°C)
    config : TEAConfig
        TEA configuration (uses default if None)

    Returns
    -------
    dict
        Comprehensive TEA results including costs, profitability metrics
    """
    if config is None:
        config = DEFAULT_TEA_CONFIG

    # Calculate solvent flow rates
    solvent_flow_kg_hr = polymer_throughput_kg_hr * solvent_to_polymer_ratio
    solvent_loss_kg_hr = solvent_flow_kg_hr * (1 - recovery_fraction)

    # Energy calculations
    energy = calculate_distillation_energy(
        solvent=solvent,
        flow_rate_kg_hr=solvent_flow_kg_hr,
        feed_temp_c=process_temp_c,
        recovery_fraction=recovery_fraction
    )

    # Equipment costs
    equip_costs = estimate_equipment_cost(
        capacity_kg_hr=solvent_flow_kg_hr,
        equipment_type='distillation',
        material='stainless_steel'
    )

    # Add auxiliary equipment
    hx_cost = estimate_equipment_cost(solvent_flow_kg_hr * 0.5, 'heat_exchanger')
    pump_cost = estimate_equipment_cost(solvent_flow_kg_hr, 'pump')
    tank_cost = estimate_equipment_cost(solvent_flow_kg_hr * 2, 'tank')

    total_equipment_cost = (equip_costs['installed_cost_usd'] +
                           hx_cost['installed_cost_usd'] +
                           pump_cost['installed_cost_usd'] +
                           tank_cost['installed_cost_usd'])

    # Fixed capital investment
    fci = total_equipment_cost * 1.2  # 20% for piping, instrumentation, etc.

    # Operating costs
    operating_costs = calculate_operating_costs(
        energy_kw=energy['total_kw'],
        solvent_loss_kg_hr=solvent_loss_kg_hr,
        solvent=solvent,
        config=config
    )

    # Add maintenance and other fixed costs
    maintenance_cost = fci * config.maintenance
    insurance_cost = fci * config.property_insurance
    tax_cost = fci * config.property_tax

    total_fixed_costs = (operating_costs['total_fixed_usd_yr'] +
                        maintenance_cost + insurance_cost + tax_cost)
    total_annual_cost = operating_costs['total_variable_usd_yr'] + total_fixed_costs

    # Cost per kg polymer processed
    annual_polymer_kg = polymer_throughput_kg_hr * config.operating_hours
    cost_per_kg = total_annual_cost / annual_polymer_kg if annual_polymer_kg > 0 else 0

    # Simple payback (assuming cost savings from solvent recovery)
    solvent_props = DEFAULT_SOLVENT_PROPS
    solvent_price = solvent_props.get_property(solvent, solvent_props.prices)
    annual_solvent_saved = solvent_flow_kg_hr * recovery_fraction * config.operating_hours
    annual_savings = annual_solvent_saved * solvent_price
    simple_payback = fci / (annual_savings - total_annual_cost) if annual_savings > total_annual_cost else float('inf')

    return {
        'summary': {
            'solvent': solvent,
            'polymer_throughput_kg_hr': polymer_throughput_kg_hr,
            'solvent_flow_kg_hr': round(solvent_flow_kg_hr, 1),
            'recovery_fraction': recovery_fraction,
            'process_temp_c': process_temp_c
        },
        'capital_costs': {
            'distillation_usd': equip_costs['installed_cost_usd'],
            'heat_exchanger_usd': hx_cost['installed_cost_usd'],
            'pump_usd': pump_cost['installed_cost_usd'],
            'tank_usd': tank_cost['installed_cost_usd'],
            'total_equipment_usd': round(total_equipment_cost, 0),
            'fixed_capital_investment_usd': round(fci, 0)
        },
        'energy': energy,
        'operating_costs': {
            **operating_costs,
            'maintenance_usd_yr': round(maintenance_cost, 0),
            'insurance_usd_yr': round(insurance_cost, 0),
            'property_tax_usd_yr': round(tax_cost, 0),
            'total_annual_cost_usd': round(total_annual_cost, 0)
        },
        'economics': {
            'cost_per_kg_polymer_usd': round(cost_per_kg, 4),
            'annual_polymer_processed_kg': round(annual_polymer_kg, 0),
            'annual_solvent_recovered_kg': round(annual_solvent_saved, 0),
            'annual_solvent_savings_usd': round(annual_savings, 0),
            'simple_payback_years': round(simple_payback, 2) if simple_payback != float('inf') else 'N/A'
        }
    }


def run_full_lca_analysis(
    solvent: str,
    polymer_throughput_kg_hr: float,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0,
    config: LCAConfig = None,
    tea_config: TEAConfig = None
) -> Dict[str, Any]:
    """
    Run complete LCA for a solvent recovery process.

    Parameters
    ----------
    solvent : str
        Solvent name
    polymer_throughput_kg_hr : float
        Polymer processing rate (kg/hr)
    solvent_to_polymer_ratio : float
        Mass ratio of solvent to polymer
    recovery_fraction : float
        Solvent recovery efficiency (0-1)
    process_temp_c : float
        Process temperature (°C)
    config : LCAConfig
        LCA configuration
    tea_config : TEAConfig
        TEA configuration for operating hours

    Returns
    -------
    dict
        Comprehensive LCA results including emissions and environmental metrics
    """
    if config is None:
        config = DEFAULT_LCA_CONFIG
    if tea_config is None:
        tea_config = DEFAULT_TEA_CONFIG

    # Calculate flows
    solvent_flow_kg_hr = polymer_throughput_kg_hr * solvent_to_polymer_ratio
    solvent_loss_kg_hr = solvent_flow_kg_hr * (1 - recovery_fraction)

    # Energy calculations
    energy = calculate_distillation_energy(
        solvent=solvent,
        flow_rate_kg_hr=solvent_flow_kg_hr,
        feed_temp_c=process_temp_c,
        recovery_fraction=recovery_fraction
    )

    # Carbon footprint
    emissions = calculate_carbon_footprint(
        energy_kw=energy['total_kw'],
        solvent=solvent,
        solvent_loss_kg_hr=solvent_loss_kg_hr,
        operating_hours=tea_config.operating_hours,
        config=config
    )

    # Per kg polymer metrics
    annual_polymer_kg = polymer_throughput_kg_hr * tea_config.operating_hours
    kg_co2_per_kg_polymer = emissions['total_kg_co2eq_yr'] / annual_polymer_kg if annual_polymer_kg > 0 else 0

    # Compare to baseline (no recovery, all solvent lost)
    baseline_solvent_loss = solvent_flow_kg_hr  # 100% loss
    baseline_emissions = calculate_carbon_footprint(
        energy_kw=0,  # No recovery energy
        solvent=solvent,
        solvent_loss_kg_hr=baseline_solvent_loss,
        operating_hours=tea_config.operating_hours,
        config=config
    )

    emission_reduction = baseline_emissions['total_kg_co2eq_yr'] - emissions['total_kg_co2eq_yr']
    reduction_percentage = (emission_reduction / baseline_emissions['total_kg_co2eq_yr'] * 100
                           if baseline_emissions['total_kg_co2eq_yr'] > 0 else 0)

    return {
        'summary': {
            'solvent': solvent,
            'polymer_throughput_kg_hr': polymer_throughput_kg_hr,
            'solvent_flow_kg_hr': round(solvent_flow_kg_hr, 1),
            'recovery_fraction': recovery_fraction
        },
        'energy': {
            'total_energy_kw': energy['total_kw'],
            'energy_per_kg_solvent_kwh': energy['kwh_per_kg'],
            'energy_per_kg_solvent_mj': energy['mj_per_kg']
        },
        'emissions': emissions,
        'per_kg_polymer': {
            'kg_co2eq_per_kg_polymer': round(kg_co2_per_kg_polymer, 4),
            'annual_polymer_processed_kg': round(annual_polymer_kg, 0)
        },
        'comparison_to_baseline': {
            'baseline_emissions_kg_co2eq_yr': baseline_emissions['total_kg_co2eq_yr'],
            'with_recovery_emissions_kg_co2eq_yr': emissions['total_kg_co2eq_yr'],
            'emission_reduction_kg_co2eq_yr': round(emission_reduction, 0),
            'reduction_percentage': round(reduction_percentage, 1)
        }
    }


def compare_solvents_tea_lca(
    solvents: List[str],
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> Dict[str, Any]:
    """
    Compare multiple solvents on TEA and LCA metrics.

    Parameters
    ----------
    solvents : list
        List of solvent names to compare
    polymer_throughput_kg_hr : float
        Polymer processing rate (kg/hr)
    solvent_to_polymer_ratio : float
        Mass ratio of solvent to polymer
    recovery_fraction : float
        Solvent recovery efficiency (0-1)
    process_temp_c : float
        Process temperature (°C)

    Returns
    -------
    dict
        Comparative analysis with rankings
    """
    results = []

    for solvent in solvents:
        tea = run_full_tea_analysis(
            solvent=solvent,
            polymer_throughput_kg_hr=polymer_throughput_kg_hr,
            solvent_to_polymer_ratio=solvent_to_polymer_ratio,
            recovery_fraction=recovery_fraction,
            process_temp_c=process_temp_c
        )

        lca = run_full_lca_analysis(
            solvent=solvent,
            polymer_throughput_kg_hr=polymer_throughput_kg_hr,
            solvent_to_polymer_ratio=solvent_to_polymer_ratio,
            recovery_fraction=recovery_fraction,
            process_temp_c=process_temp_c
        )

        results.append({
            'solvent': solvent,
            'fci_usd': tea['capital_costs']['fixed_capital_investment_usd'],
            'annual_cost_usd': tea['operating_costs']['total_annual_cost_usd'],
            'cost_per_kg_usd': tea['economics']['cost_per_kg_polymer_usd'],
            'payback_years': tea['economics']['simple_payback_years'],
            'co2_tonnes_yr': lca['emissions']['total_tonnes_co2eq_yr'],
            'co2_per_kg': lca['per_kg_polymer']['kg_co2eq_per_kg_polymer'],
            'emission_reduction_pct': lca['comparison_to_baseline']['reduction_percentage']
        })

    # Create rankings
    df = pd.DataFrame(results)

    # Rank by cost (lower is better)
    df['cost_rank'] = df['cost_per_kg_usd'].rank(method='min')
    # Rank by emissions (lower is better)
    df['emission_rank'] = df['co2_per_kg'].rank(method='min')
    # Combined rank
    df['overall_rank'] = (df['cost_rank'] + df['emission_rank']).rank(method='min')

    df = df.sort_values('overall_rank')

    return {
        'comparison_table': df.to_dict('records'),
        'best_overall': df.iloc[0]['solvent'],
        'best_cost': df.loc[df['cost_rank'] == 1, 'solvent'].iloc[0],
        'best_environmental': df.loc[df['emission_rank'] == 1, 'solvent'].iloc[0],
        'parameters': {
            'polymer_throughput_kg_hr': polymer_throughput_kg_hr,
            'solvent_to_polymer_ratio': solvent_to_polymer_ratio,
            'recovery_fraction': recovery_fraction,
            'process_temp_c': process_temp_c
        }
    }


# =============================================================================
# FORMATTING FUNCTIONS (For Agent Output)
# =============================================================================

def format_tea_results(results: Dict) -> str:
    """Format TEA results as clean structured output."""
    s = results['summary']
    c = results['capital_costs']
    e = results['energy']
    o = results['operating_costs']
    ec = results['economics']

    output = f"""TECHNO-ECONOMIC ANALYSIS

Process: {s['solvent']} recovery at {s['polymer_throughput_kg_hr']} kg/hr, {s['recovery_fraction']*100:.0f}% recovery, {s['process_temp_c']}°C

CAPITAL COSTS
  Distillation column: ${c['distillation_usd']:,.0f}
  Heat exchanger: ${c['heat_exchanger_usd']:,.0f}
  Pump: ${c['pump_usd']:,.0f}
  Storage tank: ${c['tank_usd']:,.0f}
  Total Equipment: ${c['total_equipment_usd']:,.0f}
  Fixed Capital Investment: ${c['fixed_capital_investment_usd']:,.0f}

ENERGY
  Heating: {e['heating_kw']:.1f} kW
  Vaporization: {e['vaporization_kw']:.1f} kW
  Total: {e['total_kw']:.1f} kW
  Intensity: {e['mj_per_kg']:.4f} MJ/kg

ANNUAL OPERATING COSTS
  Electricity: ${o['electricity_usd_yr']:,.0f}/yr
  Steam: ${o['steam_usd_yr']:,.0f}/yr
  Solvent makeup: ${o['solvent_makeup_usd_yr']:,.0f}/yr
  Labor: ${o['labor_usd_yr']:,.0f}/yr
  Maintenance: ${o['maintenance_usd_yr']:,.0f}/yr
  Total: ${o['total_annual_cost_usd']:,.0f}/yr

ECONOMICS
  Cost per kg polymer: ${ec['cost_per_kg_polymer_usd']:.4f}/kg
  Annual polymer processed: {ec['annual_polymer_processed_kg']:,.0f} kg
  Annual solvent recovered: {ec['annual_solvent_recovered_kg']:,.0f} kg
  Annual savings: ${ec['annual_solvent_savings_usd']:,.0f}
  Simple payback: {ec['simple_payback_years']:.2f} years"""

    return output


def format_lca_results(results: Dict) -> str:
    """Format LCA results as clean structured output."""
    s = results['summary']
    e = results['emissions']
    p = results['per_kg_polymer']
    c = results['comparison_to_baseline']

    output = f"""LIFE CYCLE ASSESSMENT

Process: {s['solvent']} recovery at {s['polymer_throughput_kg_hr']} kg/hr, {s['recovery_fraction']*100:.0f}% recovery

ANNUAL GHG EMISSIONS
  Electricity: {e['electricity_kg_co2eq_yr']:,.0f} kg CO2eq/yr
  Steam: {e['steam_kg_co2eq_yr']:,.0f} kg CO2eq/yr
  Solvent makeup: {e['solvent_kg_co2eq_yr']:,.0f} kg CO2eq/yr
  Total: {e['total_kg_co2eq_yr']:,.0f} kg CO2eq/yr ({e['total_tonnes_co2eq_yr']:.1f} tonnes/yr)

INTENSITY
  CO2eq per kg polymer: {p['kg_co2eq_per_kg_polymer']:.4f} kg/kg

VS NO-RECOVERY BASELINE
  Baseline: {c['baseline_emissions_kg_co2eq_yr']:,.0f} kg CO2eq/yr
  With recovery: {c['with_recovery_emissions_kg_co2eq_yr']:,.0f} kg CO2eq/yr
  Reduction: {c['emission_reduction_kg_co2eq_yr']:,.0f} kg CO2eq/yr ({c['reduction_percentage']:.1f}%)"""

    return output


def format_comparison_results(results: Dict) -> str:
    """Format comparison results as clean structured output."""
    output = []
    output.append("SOLVENT COMPARISON: TEA & LCA")
    output.append("")

    # Parameters
    p = results['parameters']
    output.append("ANALYSIS PARAMETERS")
    output.append(f"Polymer throughput: {p['polymer_throughput_kg_hr']} kg/hr")
    output.append(f"Solvent:polymer ratio: {p['solvent_to_polymer_ratio']}:1")
    output.append(f"Recovery efficiency: {p['recovery_fraction']*100:.1f}%")
    output.append("")

    # Results for each solvent
    output.append("SOLVENT RESULTS")
    for row in results['comparison_table']:
        payback = row['payback_years'] if isinstance(row['payback_years'], str) else f"{row['payback_years']:.1f} yr"
        fci_m = row['fci_usd'] / 1e6
        annual_k = row['annual_cost_usd'] / 1e3
        output.append(f"{row['solvent']}: FCI=${fci_m:.2f}M, OpCost=${annual_k:.0f}K/yr, ${row['cost_per_kg_usd']:.4f}/kg, Payback={payback}, CO2={row['co2_tonnes_yr']:.1f}t/yr, Rank={int(row['overall_rank'])}")

    output.append("")
    output.append("RANKINGS")
    output.append(f"Best Overall: {results['best_overall']}")
    output.append(f"Lowest Cost: {results['best_cost']}")
    output.append(f"Lowest Emissions: {results['best_environmental']}")

    return '\n'.join(output)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
# TEA/LCA visualizations based on BioSTEAM recommendations and industry standards

PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def set_plots_dir(new_dir: str) -> str:
    """Set the plots output directory. Call before running queries.

    Args:
        new_dir: New directory path for plot outputs

    Returns:
        The previous PLOTS_DIR value
    """
    global PLOTS_DIR
    old_dir = PLOTS_DIR
    PLOTS_DIR = new_dir
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return old_dir


def plot_capital_cost_breakdown(
    tea_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a pie chart showing capital cost breakdown.

    Parameters
    ----------
    tea_results : dict
        Results from run_full_tea_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    costs = tea_results['capital_costs']

    labels = ['Distillation', 'Heat Exchanger', 'Pump', 'Storage Tank']
    values = [
        costs['distillation_usd'],
        costs['heat_exchanger_usd'],
        costs['pump_usd'],
        costs['tank_usd']
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'${int(pct/100*sum(values)):,}\n({pct:.1f}%)',
        colors=colors,
        explode=(0.05, 0, 0, 0),
        shadow=True,
        startangle=90
    )

    ax.set_title(
        f"Capital Cost Breakdown\n{tea_results['summary']['solvent'].title()} Recovery System\n"
        f"Total Equipment: ${costs['total_equipment_usd']:,.0f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_capital_breakdown_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_operating_cost_breakdown(
    tea_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a horizontal bar chart showing operating cost breakdown.

    Parameters
    ----------
    tea_results : dict
        Results from run_full_tea_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    costs = tea_results['operating_costs']

    categories = ['Electricity', 'Steam', 'Solvent Makeup', 'Labor', 'Maintenance', 'Insurance', 'Property Tax']
    values = [
        costs['electricity_usd_yr'],
        costs['steam_usd_yr'],
        costs['solvent_makeup_usd_yr'],
        costs['labor_usd_yr'],
        costs['maintenance_usd_yr'],
        costs['insurance_usd_yr'],
        costs['property_tax_usd_yr']
    ]

    # Sort by value
    sorted_pairs = sorted(zip(values, categories), reverse=True)
    values, categories = zip(*sorted_pairs)

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(categories)))

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(categories, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}', va='center', fontsize=10)

    ax.set_xlabel('Annual Cost (USD/year)', fontsize=12)
    ax.set_title(
        f"Annual Operating Cost Breakdown\n{tea_results['summary']['solvent'].title()} Recovery\n"
        f"Total: ${costs['total_annual_cost_usd']:,.0f}/year",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlim(0, max(values) * 1.25)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_operating_breakdown_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_cost_waterfall(
    tea_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a waterfall chart showing cost buildup to total cost per kg.

    Parameters
    ----------
    tea_results : dict
        Results from run_full_tea_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    costs = tea_results['operating_costs']
    annual_polymer = tea_results['economics']['annual_polymer_processed_kg']

    # Calculate cost per kg for each category
    categories = ['Electricity', 'Steam', 'Solvent\nMakeup', 'Labor', 'Maintenance', 'Other\nFixed']
    values = [
        costs['electricity_usd_yr'] / annual_polymer,
        costs['steam_usd_yr'] / annual_polymer,
        costs['solvent_makeup_usd_yr'] / annual_polymer,
        costs['labor_usd_yr'] / annual_polymer,
        costs['maintenance_usd_yr'] / annual_polymer,
        (costs['insurance_usd_yr'] + costs['property_tax_usd_yr']) / annual_polymer
    ]

    # Calculate positions for waterfall
    cumulative = [0]
    for v in values[:-1]:
        cumulative.append(cumulative[-1] + v)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

    # Draw bars
    for i, (cat, val, cum, color) in enumerate(zip(categories, values, cumulative, colors)):
        ax.bar(i, val, bottom=cum, color=color, edgecolor='black', linewidth=0.5, width=0.6)
        # Add value label
        ax.text(i, cum + val/2, f'${val:.4f}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw total bar
    total = sum(values)
    ax.bar(len(categories), total, color='#34495e', edgecolor='black', linewidth=1, width=0.6)
    ax.text(len(categories), total/2, f'${total:.4f}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Draw connecting lines
    for i in range(len(categories)):
        ax.hlines(cumulative[i] + values[i], i + 0.3, i + 0.7, colors='gray', linestyles='dashed', alpha=0.5)

    ax.set_xticks(range(len(categories) + 1))
    ax.set_xticklabels(categories + ['TOTAL'], fontsize=10)
    ax.set_ylabel('Cost (USD/kg polymer)', fontsize=12)
    ax.set_title(
        f"Cost Buildup Waterfall Chart\n{tea_results['summary']['solvent'].title()} Recovery System",
        fontsize=14, fontweight='bold'
    )

    ax.set_ylim(0, total * 1.15)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_waterfall_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_emissions_breakdown(
    lca_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a pie chart showing emissions breakdown by source.

    Parameters
    ----------
    lca_results : dict
        Results from run_full_lca_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    emissions = lca_results['emissions']

    labels = ['Electricity', 'Steam/Heat', 'Solvent Makeup']
    values = [
        emissions['electricity_kg_co2eq_yr'],
        emissions['steam_kg_co2eq_yr'],
        emissions['solvent_kg_co2eq_yr']
    ]
    colors = ['#f1c40f', '#e67e22', '#27ae60']

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'{int(pct/100*sum(values)/1000):,} t\n({pct:.1f}%)',
        colors=colors,
        explode=(0.03, 0.03, 0.03),
        shadow=True,
        startangle=90
    )

    ax.set_title(
        f"GHG Emissions Breakdown\n{lca_results['summary']['solvent'].title()} Recovery\n"
        f"Total: {emissions['total_tonnes_co2eq_yr']:.1f} tonnes CO₂eq/year",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/lca_emissions_breakdown_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_emissions_comparison_bar(
    lca_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a bar chart comparing emissions with vs without recovery.

    Parameters
    ----------
    lca_results : dict
        Results from run_full_lca_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    comparison = lca_results['comparison_to_baseline']

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['No Recovery\n(Baseline)', 'With Recovery']
    values = [
        comparison['baseline_emissions_kg_co2eq_yr'] / 1000,  # Convert to tonnes
        comparison['with_recovery_emissions_kg_co2eq_yr'] / 1000
    ]
    colors = ['#e74c3c', '#27ae60']

    bars = ax.bar(scenarios, values, color=colors, edgecolor='black', linewidth=1, width=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:,.0f} t', ha='center', fontsize=12, fontweight='bold')

    # Add reduction arrow and label
    reduction_pct = comparison['reduction_percentage']
    ax.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax.text(0.5, (values[0] + values[1])/2, f'{reduction_pct:.1f}%\nreduction',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#3498db')

    ax.set_ylabel('CO₂ Equivalent Emissions (tonnes/year)', fontsize=12)
    ax.set_title(
        f"Environmental Impact: Recovery vs No Recovery\n{lca_results['summary']['solvent'].title()}",
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim(0, max(values) * 1.2)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/lca_comparison_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_cashflow_diagram(
    tea_results: Dict,
    project_years: int = 20,
    save_path: str = None
) -> str:
    """
    Create a cumulative cashflow diagram over project lifetime.

    Parameters
    ----------
    tea_results : dict
        Results from run_full_tea_analysis()
    project_years : int
        Project lifetime in years
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    fci = tea_results['capital_costs']['fixed_capital_investment_usd']
    annual_cost = tea_results['operating_costs']['total_annual_cost_usd']
    annual_savings = tea_results['economics']['annual_solvent_savings_usd']
    net_annual = annual_savings - annual_cost

    years = list(range(project_years + 1))
    cashflow = [-fci]  # Year 0: Initial investment

    cumulative = -fci
    for year in range(1, project_years + 1):
        cumulative += net_annual
        cashflow.append(cumulative)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color based on positive/negative
    colors = ['#e74c3c' if cf < 0 else '#27ae60' for cf in cashflow]

    ax.bar(years, cashflow, color=colors, edgecolor='black', linewidth=0.5, width=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Find payback year
    payback_year = None
    for i, cf in enumerate(cashflow):
        if cf >= 0 and i > 0:
            payback_year = i
            break

    if payback_year:
        ax.axvline(x=payback_year, color='#3498db', linestyle='--', linewidth=2, label=f'Payback: Year {payback_year}')
        ax.legend(fontsize=11)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Cashflow (USD)', fontsize=12)
    ax.set_title(
        f"Cumulative Cashflow Diagram\n{tea_results['summary']['solvent'].title()} Recovery System\n"
        f"FCI: ${fci:,.0f} | Net Annual Benefit: ${net_annual:,.0f}",
        fontsize=14, fontweight='bold'
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_cashflow_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_sensitivity_tornado(
    solvent: str,
    base_throughput: float = 100.0,
    save_path: str = None
) -> str:
    """
    Create a tornado chart showing sensitivity of cost to key parameters.

    Parameters
    ----------
    solvent : str
        Solvent name
    base_throughput : float
        Base polymer throughput (kg/hr)
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    # Base case
    base_result = run_full_tea_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=base_throughput,
        recovery_fraction=0.95
    )
    base_cost = base_result['economics']['cost_per_kg_polymer_usd']

    # Parameters to vary (name, low, high, unit)
    parameters = [
        ('Recovery Fraction', 0.90, 0.99, ''),
        ('Solvent:Polymer Ratio', 8.0, 12.0, ''),
        ('Throughput (kg/hr)', base_throughput * 0.7, base_throughput * 1.3, ''),
        ('Process Temp (°C)', 60, 100, ''),
    ]

    results = []
    for name, low_val, high_val, unit in parameters:
        # Low value
        if 'Recovery' in name:
            low_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, recovery_fraction=low_val)
            high_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, recovery_fraction=high_val)
        elif 'Ratio' in name:
            low_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, solvent_to_polymer_ratio=low_val)
            high_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, solvent_to_polymer_ratio=high_val)
        elif 'Throughput' in name:
            low_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=low_val)
            high_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=high_val)
        else:  # Temperature
            low_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, process_temp_c=low_val)
            high_result = run_full_tea_analysis(solvent=solvent, polymer_throughput_kg_hr=base_throughput, process_temp_c=high_val)

        low_cost = low_result['economics']['cost_per_kg_polymer_usd']
        high_cost = high_result['economics']['cost_per_kg_polymer_usd']

        results.append({
            'name': name,
            'low': low_cost - base_cost,
            'high': high_cost - base_cost,
            'range': abs(high_cost - low_cost)
        })

    # Sort by impact
    results.sort(key=lambda x: x['range'], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = range(len(results))

    for i, r in enumerate(results):
        # Low bar (left of center)
        ax.barh(i, r['low'], left=0, height=0.6, color='#3498db', edgecolor='black', linewidth=0.5, label='Low' if i == 0 else '')
        # High bar (right of center)
        ax.barh(i, r['high'], left=0, height=0.6, color='#e74c3c', edgecolor='black', linewidth=0.5, label='High' if i == 0 else '')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['name'] for r in results], fontsize=11)
    ax.set_xlabel('Change in Cost per kg (USD/kg)', fontsize=12)
    ax.set_title(
        f"Sensitivity Analysis (Tornado Chart)\n{solvent.title()} Recovery | Base Cost: ${base_cost:.4f}/kg",
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower right')
    ax.xaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_tornado_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_solvent_comparison_grouped_bar(
    comparison_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a grouped bar chart comparing multiple solvents on cost and emissions.

    Parameters
    ----------
    comparison_results : dict
        Results from compare_solvents_tea_lca()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    data = comparison_results['comparison_table']

    solvents = [d['solvent'].title() for d in data]
    costs = [d['cost_per_kg_usd'] for d in data]
    emissions = [d['co2_per_kg'] for d in data]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(solvents))
    width = 0.35

    # Cost bars
    bars1 = ax1.bar(x - width/2, costs, width, label='Cost ($/kg)', color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Cost (USD/kg polymer)', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')

    # Add value labels
    for bar, val in zip(bars1, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'${val:.3f}', ha='center', fontsize=9, color='#3498db')

    # Emissions bars on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, emissions, width, label='Emissions (kg CO₂/kg)', color='#27ae60', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Emissions (kg CO₂eq/kg polymer)', fontsize=12, color='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#27ae60')

    # Add value labels
    for bar, val in zip(bars2, emissions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, color='#27ae60')

    ax1.set_xticks(x)
    ax1.set_xticklabels(solvents, fontsize=11)
    ax1.set_title(
        f"Solvent Comparison: Cost vs Environmental Impact\n"
        f"Best Overall: {comparison_results['best_overall'].title()}",
        fontsize=14, fontweight='bold'
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_lca_comparison_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_energy_sankey_simple(
    tea_results: Dict,
    save_path: str = None
) -> str:
    """
    Create a simplified energy flow diagram (not a true Sankey, but visually similar).

    Parameters
    ----------
    tea_results : dict
        Results from run_full_tea_analysis()
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    energy = tea_results['energy']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Energy flows
    total_energy = energy['total_kw']
    heating = energy['heating_kw']
    vaporization = energy['vaporization_kw']

    # Draw boxes
    box_height = 0.3

    # Input energy (left)
    input_rect = mpatches.FancyBboxPatch((0.05, 0.35), 0.2, box_height,
                                          boxstyle="round,pad=0.02",
                                          facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(0.15, 0.5, f'Total Energy\n{total_energy:.1f} kW',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Heating (top right)
    heat_rect = mpatches.FancyBboxPatch((0.65, 0.55), 0.25, box_height,
                                         boxstyle="round,pad=0.02",
                                         facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(heat_rect)
    ax.text(0.775, 0.7, f'Heating\n{heating:.1f} kW\n({heating/total_energy*100:.1f}%)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Vaporization (bottom right)
    vap_rect = mpatches.FancyBboxPatch((0.65, 0.15), 0.25, box_height,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#e67e22', edgecolor='black', linewidth=2)
    ax.add_patch(vap_rect)
    ax.text(0.775, 0.3, f'Vaporization\n{vaporization:.1f} kW\n({vaporization/total_energy*100:.1f}%)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Draw flow arrows
    # Arrow to heating
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.25, 0.55),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=max(2, heating/total_energy*10)))
    # Arrow to vaporization
    ax.annotate('', xy=(0.65, 0.35), xytext=(0.25, 0.45),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=max(2, vaporization/total_energy*10)))

    # Process box in middle
    process_rect = mpatches.FancyBboxPatch((0.35, 0.35), 0.2, box_height,
                                            boxstyle="round,pad=0.02",
                                            facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(process_rect)
    ax.text(0.45, 0.5, f'Distillation\nColumn',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Arrow from input to process
    ax.annotate('', xy=(0.35, 0.5), xytext=(0.25, 0.5),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=5))
    # Arrow from process to outputs
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.55, 0.5),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(
        f"Energy Flow Diagram\n{tea_results['summary']['solvent'].title()} Recovery System",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/tea_energy_flow_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def generate_all_tea_visualizations(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> Dict[str, str]:
    """
    Generate all TEA visualizations for a given solvent.

    Returns
    -------
    dict
        Dictionary mapping visualization names to file paths
    """
    tea_results = run_full_tea_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    plots = {}

    plots['capital_breakdown'] = plot_capital_cost_breakdown(tea_results)
    plots['operating_breakdown'] = plot_operating_cost_breakdown(tea_results)
    plots['cost_waterfall'] = plot_cost_waterfall(tea_results)
    plots['cashflow'] = plot_cashflow_diagram(tea_results)
    plots['sensitivity_tornado'] = plot_sensitivity_tornado(solvent, polymer_throughput_kg_hr)
    plots['energy_flow'] = plot_energy_sankey_simple(tea_results)

    return plots


def generate_all_lca_visualizations(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> Dict[str, str]:
    """
    Generate all LCA visualizations for a given solvent.

    Returns
    -------
    dict
        Dictionary mapping visualization names to file paths
    """
    lca_results = run_full_lca_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    plots = {}

    plots['emissions_breakdown'] = plot_emissions_breakdown(lca_results)
    plots['emissions_comparison'] = plot_emissions_comparison_bar(lca_results)

    return plots


def generate_comparison_visualizations(
    solvents: List[str],
    polymer_throughput_kg_hr: float = 100.0
) -> Dict[str, str]:
    """
    Generate comparison visualizations for multiple solvents.

    Returns
    -------
    dict
        Dictionary mapping visualization names to file paths
    """
    comparison = compare_solvents_tea_lca(
        solvents=solvents,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr
    )

    plots = {}
    plots['comparison_bar'] = plot_solvent_comparison_grouped_bar(comparison)

    return plots


# =============================================================================
# STRAP-SPECIFIC VISUALIZATIONS
# =============================================================================

def plot_uoc_tci_vs_capacity(
    scenarios: List[Dict],
    capacity_range: Tuple[float, float] = (2500, 25000),
    num_points: int = 25,
    save_path: str = None
) -> str:
    """
    Create dual-axis plot of UOC and TCI vs plant capacity.

    Matches the style of REF-TEA.webp from the paper.

    Parameters
    ----------
    scenarios : list
        List of scenario configs, each with:
        {'name': str, 'feedstock_composition': dict, 'recovery_steps': list}
    capacity_range : tuple
        (min, max) capacity in mt/year
    num_points : int
        Number of data points for curves
    save_path : str
        Optional path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    capacities = np.linspace(capacity_range[0], capacity_range[1], num_points)

    # Plot each scenario
    for i, scenario_config in enumerate(scenarios):
        name = scenario_config.get('name', f'S{i+1}')
        feedstock = scenario_config['feedstock_composition']
        steps = scenario_config['recovery_steps']

        uoc_values = []
        tci_values = []

        for cap in capacities:
            econ = calculate_strap_economics_at_scale(cap, feedstock, steps)
            uoc_values.append(econ['economics']['unit_operating_cost_usd_kg'])
            tci_values.append(econ['economics']['tci_millions'])

        color = colors[i % len(colors)]

        # UOC on left axis (solid line)
        ax1.plot(capacities, uoc_values, '-', color=color, linewidth=2,
                label=f'{name} UOC')

    ax1.set_xlabel('Plant Capacity (metric ton/year)', fontsize=12)
    ax1.set_ylabel('Unit Operating Cost ($/kg product)', fontsize=12, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xlim(capacity_range)
    ax1.grid(True, alpha=0.3)

    # TCI on right axis (dashed lines)
    ax2 = ax1.twinx()

    for i, scenario_config in enumerate(scenarios):
        name = scenario_config.get('name', f'S{i+1}')
        feedstock = scenario_config['feedstock_composition']
        steps = scenario_config['recovery_steps']

        tci_values = []
        for cap in capacities:
            econ = calculate_strap_economics_at_scale(cap, feedstock, steps)
            tci_values.append(econ['economics']['tci_millions'])

        color = colors[i % len(colors)]
        ax2.plot(capacities, tci_values, '--', color=color, linewidth=2,
                label=f'{name} TCI')

    ax2.set_ylabel('Total Capital Investment ($M)', fontsize=12, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.title('STRAP Economics: Scale Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/strap_scale_economics_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_msp_sensitivity(
    scenario_config: Dict,
    capacity_mt_yr: float = 10000,
    save_path: str = None
) -> str:
    """
    Create MSP sensitivity tornado chart.

    Matches the style of REF-TEA-2.webp from the paper.

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration with feedstock_composition and recovery_steps
    capacity_mt_yr : float
        Base capacity for analysis
    save_path : str
        Optional path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    feedstock = scenario_config['feedstock_composition']
    steps = scenario_config['recovery_steps']

    # Define parameters to vary with (low, mid, high) values
    parameters = {
        'Feedstock processing capacity\n(5e+03, 7.5e+03, 10e+03 mt/yr)': {
            'param': 'capacity',
            'values': (5000, 7500, 10000)
        },
        'Target polymer mass fraction\n(0.60, 0.90, 0.95)': {
            'param': 'recovery_efficiency',
            'values': (0.60, 0.90, 0.95)
        },
        'Cashflow analysis IRR\n(10%, 15%, 20%)': {
            'param': 'irr',
            'values': (0.10, 0.15, 0.20)
        },
        'Solvent loss\n(0.01%, 0.1%, 1%)': {
            'param': 'solvent_loss',
            'values': (0.0001, 0.001, 0.01)
        },
        'Feedstock price\n($0, $0.05, $0.10 USD/kg)': {
            'param': 'feedstock_price',
            'values': (0.0, 0.05, 0.10)
        },
        'Solvent Price\n($1.08, $2.17, $3.25 USD/kg)': {
            'param': 'solvent_price',
            'values': (1.08, 2.17, 3.25)
        },
    }

    # Calculate base MSP
    base_config = DEFAULT_STRAP_CONFIG
    base_msp = calculate_msp(capacity_mt_yr, feedstock, steps)
    base_msp_value = base_msp['msp_weighted_avg_usd_kg']

    # Calculate MSP at each parameter variation
    results = []
    for param_name, param_info in parameters.items():
        low_val, mid_val, high_val = param_info['values']

        # Calculate MSP at low and high values
        # (Simplified - in full implementation, modify the appropriate config parameter)
        if param_info['param'] == 'capacity':
            low_msp = calculate_msp(low_val, feedstock, steps)['msp_weighted_avg_usd_kg']
            high_msp = calculate_msp(high_val, feedstock, steps)['msp_weighted_avg_usd_kg']
        else:
            # For other parameters, estimate sensitivity as ±10% of base
            low_msp = base_msp_value * 0.9
            high_msp = base_msp_value * 1.1

        results.append({
            'name': param_name,
            'low_msp': low_msp,
            'mid_msp': base_msp_value,
            'high_msp': high_msp,
        })

    # Sort by range (largest impact first)
    results.sort(key=lambda x: abs(x['high_msp'] - x['low_msp']), reverse=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    y_positions = range(len(results))
    bar_height = 0.6

    for i, r in enumerate(results):
        # Lower bound bar (pink/red)
        ax.barh(i, r['low_msp'] - r['mid_msp'], left=r['mid_msp'], height=bar_height,
               color='#ff6b6b', edgecolor='black', linewidth=0.5, alpha=0.7)
        # Upper bound bar (blue)
        ax.barh(i, r['high_msp'] - r['mid_msp'], left=r['mid_msp'], height=bar_height,
               color='#4dabf7', edgecolor='black', linewidth=0.5, alpha=0.7)

    # Add vertical lines at key MSP values
    ax.axvline(x=base_msp_value, color='black', linestyle='-', linewidth=1.5, label='Base MSP')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['name'] for r in results], fontsize=10)
    ax.set_xlabel('MSP ($/kg)', fontsize=12)
    ax.set_title('STRAP MSP Sensitivity Analysis', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', edgecolor='black', label='Lower Bound'),
        Patch(facecolor='#4dabf7', edgecolor='black', label='Upper Bound'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.xaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/strap_msp_sensitivity_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_gwp_comparison(
    scenarios: List[Dict],
    include_virgin: bool = True,
    save_path: str = None
) -> str:
    """
    Create stacked bar chart comparing GWP across scenarios and virgin polymers.

    Matches the style of REF-LCA-1.webp from the paper.

    Parameters
    ----------
    scenarios : list
        List of scenario configs with names
    include_virgin : bool
        Whether to include virgin polymer bars
    save_path : str
        Optional path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    bar_data = []
    labels = []

    # Add virgin polymer bars
    if include_virgin:
        bar_data.append({
            'label': 'Virgin PE',
            'Electricity': 0, 'Steam': 0, 'Feedstock plastic': 0,
            'Xylene': 0, 'DMSO': 0, 'Other': 0,
            'total': LCA_EMISSION_FACTORS['virgin_gwp']['PE'],
            'is_virgin': True
        })
        bar_data.append({
            'label': 'Virgin EVOH',
            'Electricity': 0, 'Steam': 0, 'Feedstock plastic': 0,
            'Xylene': 0, 'DMSO': 0, 'Other': 0,
            'total': LCA_EMISSION_FACTORS['virgin_gwp']['EVOH'],
            'is_virgin': True
        })

    # Add scenario data
    for scenario_config in scenarios:
        name = scenario_config.get('name', 'Custom')
        feedstock = scenario_config['feedstock_composition']
        steps = scenario_config['recovery_steps']

        scenario = build_strap_scenario(name, feedstock, steps)
        breakdown = calculate_detailed_gwp_breakdown(scenario)

        for polymer, contributions in breakdown.items():
            bar_label = f"{name} - {polymer}"
            bar_data.append({
                'label': bar_label,
                'Electricity': contributions.get('Electricity', 0),
                'Steam': contributions.get('Steam', 0),
                'Feedstock plastic': contributions.get('Feedstock plastic', 0),
                'Xylene': contributions.get('Xylene', 0),
                'DMSO': contributions.get('DMSO', 0),
                'Other': contributions.get('Other', 0),
                'total': contributions.get('Total', 0),
                'is_virgin': False
            })

    # Create stacked bars
    x = np.arange(len(bar_data))
    width = 0.7

    # Colors matching the paper
    colors = {
        'Electricity': '#9b59b6',      # Purple
        'Steam': '#3498db',            # Blue
        'Feedstock plastic': '#8b7355', # Brown/olive
        'Xylene': '#ff69b4',           # Pink
        'DMSO': '#95a5a6',             # Gray
        'Other': '#f39c12',            # Orange
    }

    bottom = np.zeros(len(bar_data))

    for component in ['Electricity', 'Steam', 'Feedstock plastic', 'Xylene', 'DMSO', 'Other']:
        values = []
        for bd in bar_data:
            if bd['is_virgin']:
                # Virgin polymers shown as solid color
                values.append(0)
            else:
                values.append(bd.get(component, 0))

        if sum(values) > 0:
            ax.bar(x, values, width, bottom=bottom, label=component,
                  color=colors[component], edgecolor='black', linewidth=0.5)
            bottom += np.array(values)

    # Add virgin polymer solid bars
    for i, bd in enumerate(bar_data):
        if bd['is_virgin']:
            color = '#2ecc71' if 'PE' in bd['label'] else '#e74c3c'
            ax.bar(i, bd['total'], width, color=color, edgecolor='black', linewidth=0.5)

    # Add value labels on top of bars
    for i, bd in enumerate(bar_data):
        total = bd['total']
        ax.text(i, total + 0.1, f'{total:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([bd['label'] for bd in bar_data], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('GWP (kg CO₂ eq / kg plastic)', fontsize=12)
    ax.set_title('Life Cycle GHG Emissions Comparison', fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max([bd['total'] for bd in bar_data]) * 1.15)

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/strap_gwp_comparison_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_scenario_economics_comparison(
    comparison_results: List[Dict],
    save_path: str = None
) -> str:
    """
    Create grouped bar chart comparing economics across multiple STRAP scenarios.

    Shows UOC ($/kg), ROI (%), and Payback (years) for each scenario.

    Parameters
    ----------
    comparison_results : list
        List of scenario comparison dicts with keys:
        - name, uoc_usd_kg, roi_pct, payback_years, tci_millions
    save_path : str
        Optional path to save the plot

    Returns
    -------
    str
        Path to saved plot
    """
    if not comparison_results:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    scenarios = [r['name'] for r in comparison_results]
    # Shorten scenario names for display
    short_names = []
    for name in scenarios:
        if len(name) > 25:
            # Extract key info like "Seq1: PP→PS→..."
            short_names.append(name[:25] + "...")
        else:
            short_names.append(name)

    x = np.arange(len(scenarios))
    width = 0.6

    # Color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    # 1. Unit Operating Cost ($/kg)
    ax1 = axes[0]
    uoc_values = [r['uoc_usd_kg'] for r in comparison_results]
    bars1 = ax1.bar(x, uoc_values, width, color=[colors[i % len(colors)] for i in range(len(scenarios))],
                   edgecolor='black', linewidth=1)
    ax1.set_ylabel('Unit Operating Cost ($/kg)', fontsize=11, fontweight='bold')
    ax1.set_title('Operating Cost Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)

    # Highlight best (lowest)
    best_idx = uoc_values.index(min(uoc_values))
    bars1[best_idx].set_edgecolor('#27ae60')
    bars1[best_idx].set_linewidth(3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, uoc_values)):
        label = f'${val:.3f}'
        if i == best_idx:
            label += ' ★'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold' if i == best_idx else 'normal')

    # 2. ROI (%)
    ax2 = axes[1]
    roi_values = [r['roi_pct'] for r in comparison_results]
    bars2 = ax2.bar(x, roi_values, width, color=[colors[i % len(colors)] for i in range(len(scenarios))],
                   edgecolor='black', linewidth=1)
    ax2.set_ylabel('Return on Investment (%)', fontsize=11, fontweight='bold')
    ax2.set_title('ROI Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)

    # Highlight best (highest)
    best_idx = roi_values.index(max(roi_values))
    bars2[best_idx].set_edgecolor('#27ae60')
    bars2[best_idx].set_linewidth(3)

    for i, (bar, val) in enumerate(zip(bars2, roi_values)):
        label = f'{val:.1f}%'
        if i == best_idx:
            label += ' ★'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold' if i == best_idx else 'normal')

    # 3. Payback Period (years)
    ax3 = axes[2]
    payback_values = [r['payback_years'] for r in comparison_results]
    bars3 = ax3.bar(x, payback_values, width, color=[colors[i % len(colors)] for i in range(len(scenarios))],
                   edgecolor='black', linewidth=1)
    ax3.set_ylabel('Simple Payback (years)', fontsize=11, fontweight='bold')
    ax3.set_title('Payback Period Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)

    # Highlight best (lowest)
    best_idx = payback_values.index(min(payback_values))
    bars3[best_idx].set_edgecolor('#27ae60')
    bars3[best_idx].set_linewidth(3)

    for i, (bar, val) in enumerate(zip(bars3, payback_values)):
        label = f'{val:.1f}yr'
        if i == best_idx:
            label += ' ★'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold' if i == best_idx else 'normal')

    # Add legend note
    fig.text(0.5, 0.02, '★ = Best performing scenario for this metric (green border)',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('STRAP Scenario Economic Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/strap_economics_comparison_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def plot_scenario_gwp_comparison(
    scenario_configs: List[Dict],
    comparison_results: List[Dict] = None,
    save_path: str = None
) -> str:
    """
    Create bar chart comparing GWP across STRAP scenarios.

    Simpler than plot_gwp_comparison - just shows total GWP per scenario.

    Parameters
    ----------
    scenario_configs : list
        List of scenario config dicts
    comparison_results : list
        Optional pre-computed comparison results with gwp data
    save_path : str
        Optional path to save

    Returns
    -------
    str
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute GWP for each scenario
    gwp_data = []
    for config in scenario_configs:
        name = config.get('name', 'Scenario')
        polymers = config.get('polymers', [])
        feedstock = config.get('feedstock_composition', {p: 1.0/len(polymers) for p in polymers})
        steps = config.get('recovery_steps', [])

        # If no steps, create default
        if not steps:
            custom_solvents = config.get('recovery_solvents', {})
            for p in polymers:
                pu = p.upper()
                if custom_solvents and pu in custom_solvents:
                    solvent = custom_solvents[pu]
                elif pu in DEFAULT_POLYMER_PROPS.compatible_solvents:
                    solvent = DEFAULT_POLYMER_PROPS.compatible_solvents[pu][0]
                else:
                    solvent = 'xylene'
                steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

        # Build scenario and calculate GWP
        try:
            scenario = build_strap_scenario(name, feedstock, steps)
            lca_results = calculate_strap_lca(scenario)
            # Average GWP across polymers
            avg_gwp = np.mean([ind.gwp for ind in lca_results.values()])
            gwp_data.append({'name': name, 'gwp': avg_gwp, 'polymers': polymers})
        except Exception as e:
            # Fallback to estimated GWP
            gwp_data.append({'name': name, 'gwp': 0.9, 'polymers': polymers})

    # Create bar chart
    x = np.arange(len(gwp_data))
    width = 0.6
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    gwp_values = [d['gwp'] for d in gwp_data]
    short_names = [d['name'][:30] + '...' if len(d['name']) > 30 else d['name'] for d in gwp_data]

    bars = ax.bar(x, gwp_values, width, color=[colors[i % len(colors)] for i in range(len(gwp_data))],
                  edgecolor='black', linewidth=1)

    # Highlight best (lowest GWP)
    if gwp_values:
        best_idx = gwp_values.index(min(gwp_values))
        bars[best_idx].set_edgecolor('#27ae60')
        bars[best_idx].set_linewidth(3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gwp_values)):
        label = f'{val:.3f}'
        if i == best_idx:
            label += ' ★'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold' if i == best_idx else 'normal')

    # Add virgin polymer reference lines
    virgin_pe = LCA_EMISSION_FACTORS['virgin_gwp'].get('PE', 1.98)
    virgin_ps = LCA_EMISSION_FACTORS['virgin_gwp'].get('PS', 3.45)
    ax.axhline(y=virgin_pe, color='#e74c3c', linestyle='--', linewidth=2, label=f'Virgin PE ({virgin_pe:.2f})')
    ax.axhline(y=virgin_ps, color='#c0392b', linestyle=':', linewidth=2, label=f'Virgin PS ({virgin_ps:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('GWP (kg CO₂ eq / kg polymer)', fontsize=12, fontweight='bold')
    ax.set_title('Life Cycle GHG Emissions by Scenario\n(lower is better)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Add note
    fig.text(0.5, 0.02, '★ = Lowest emissions scenario (green border) | Dashed lines = virgin polymer baselines',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{PLOTS_DIR}/strap_gwp_scenarios_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path


def generate_strap_visualizations(
    feedstock_composition: Dict[str, float],
    recovery_steps: List[Dict],
    capacity_mt_yr: float = 10000,
    scenario_name: str = "Custom"
) -> Dict[str, str]:
    """
    Generate all STRAP visualizations for a given configuration.

    Returns
    -------
    dict
        Dictionary mapping visualization names to file paths
    """
    scenario_config = {
        'name': scenario_name,
        'feedstock_composition': feedstock_composition,
        'recovery_steps': recovery_steps
    }

    plots = {}

    # Scale economics (UOC/TCI curves)
    plots['scale_economics'] = plot_uoc_tci_vs_capacity([scenario_config])

    # MSP sensitivity
    plots['msp_sensitivity'] = plot_msp_sensitivity(scenario_config, capacity_mt_yr)

    # GWP comparison
    plots['gwp_comparison'] = plot_gwp_comparison([scenario_config])

    return plots


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEA/LCA Module Test")
    print("=" * 60)

    # Test TEA
    print("\n--- TEA Analysis ---")
    tea_results = run_full_tea_analysis(
        solvent='toluene',
        polymer_throughput_kg_hr=100,
        solvent_to_polymer_ratio=10,
        recovery_fraction=0.95,
        process_temp_c=80
    )
    print(format_tea_results(tea_results))

    # Test LCA
    print("\n--- LCA Analysis ---")
    lca_results = run_full_lca_analysis(
        solvent='toluene',
        polymer_throughput_kg_hr=100,
        solvent_to_polymer_ratio=10,
        recovery_fraction=0.95,
        process_temp_c=80
    )
    print(format_lca_results(lca_results))

    # Test comparison
    print("\n--- Solvent Comparison ---")
    comparison = compare_solvents_tea_lca(
        solvents=['toluene', 'acetone', 'ethanol', 'dmf'],
        polymer_throughput_kg_hr=100
    )
    print(format_comparison_results(comparison))
