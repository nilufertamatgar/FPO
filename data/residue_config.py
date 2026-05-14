"""
Centralized residue type definitions.

All modules that need protein_residues, nucleotide_residues, or ion_residues
should import from here instead of defining their own.

Usage:
    from data.residue_config import get_protein_residues, NUCLEOTIDE, ION

    # In entry points (trainer.py, benchmark, etc.), call configure() after loading config:
    from data.residue_config import configure
    configure(include_nonstd_amino_acids=config.data.include_nonstd_amino_acids)
"""

import prody


# -- Pre-computed residue sets --
STDAA = frozenset(prody.flagDefinition("stdaa"))
NONSTDAA = frozenset(prody.flagDefinition("nonstdaa"))
NUCLEOTIDE = frozenset(prody.flagDefinition("nucleotide"))

# AF3 complete ion list (~200 types)
ION = frozenset({
    'NA6', 'OC5', 'PT', 'DSC', 'MW1', 'CUZ', 'SEK', 'OC1', 'CON', 'BEF',
    'OH', 'OCN', 'YB2', '119', 'SM', '2HP', 'E4N', 'ZO3', 'CU2', 'MN5',
    'BA', 'SO3', '3NI', 'TRA', 'NI3', 'W', 'OS4', 'NRU', 'NAO', 'BR',
    'ATH', 'EU3', 'IN', 'LA', '4TI', 'DMI', 'EU', 'NAW', 'DME', 'ZN3',
    'BSY', 'ZR', 'T1A', 'PO3', 'THE', 'NET', 'BS3', 'MW3', 'OC8', 'YT3',
    'O4M', 'ALF', 'AG', 'TBA', 'OAA', 'CUA', 'MO4', 'MO1', 'ZN2', 'RU',
    'NA5', 'PR', 'HG', 'PD', 'CU3', 'OCL', 'TCN', '4MO', 'F', 'V',
    'CU', 'BO4', '3CO', 'MOO', 'CU1', 'RHD', 'NO2', 'OS', 'TH', 'Y1',
    'TL', 'FE', 'LU', 'OF3', '118', 'KO4', '1CU', 'MH2', 'MO', 'MO6',
    'CD3', 'ZN', 'TEA', 'CD', 'AUC', 'NI2', 'CD5', 'FE2', 'OC4', 'PT4',
    'CHT', 'IR3', 'MH3', 'MO3', 'K', 'RH3', 'MAC', 'MMC', 'PBM', 'IUM',
    '2FK', 'FPO', 'TMA', '2OF', 'MW2', 'MN6', 'OCO', 'TB', 'ZCM', 'LCP',
    'CYN', 'MN3', 'YH', 'CO5', 'ZNO', 'OC6', 'AU3', 'AU', 'MG', '3OF',
    'CR', 'OC2', 'MOS', 'BF4', 'SMO', 'YB', 'MO2', 'PTN', 'EMC', 'MOW',
    'RB', 'NA2', '6MO', 'OXL', '543', 'CO', 'GD3', 'OF1', 'PB', 'VO4',
    'OC7', 'SE4', 'HAI', '3MT', 'OCM', 'VN3', 'EDR', 'MN', 'GA', 'NI',
    'OC3', 'ER3', 'DTI', 'MO5', '1AL', 'PER', 'HGC', 'SB', 'AM', 'AL',
    'LCO', 'PI', '4PU', 'WO5', 'GEP', 'HO3', 'IR', 'LI', 'CD1', 'CF',
    'IRI', 'OF2', 'CS', 'DY', 'CSB', 'NI1', 'CA', 'CAC', 'CE',
})


# -- Configurable state --
_include_nonstd = True  # default: include non-standard amino acids


def configure(include_nonstd_amino_acids=True):
    """Call from entry points after loading config."""
    global _include_nonstd
    _include_nonstd = include_nonstd_amino_acids


def get_protein_residues():
    """Return the current protein residue set based on configuration."""
    if _include_nonstd:
        return STDAA | NONSTDAA
    return set(STDAA)
