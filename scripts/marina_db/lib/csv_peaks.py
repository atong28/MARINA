"""
Shared parsers for the hand-extracted per-NPID peak CSVs in `filtered/<NPID>/`.

These were duplicated near-verbatim between build_benchmark.py and build_journal.py
(the same curation produced both benchmark sets). One copy now lives here. The header
variants below are the ones that actually occur across the hand-made files.
"""
import csv
import re


def parse_shift(val):
    """Parse a shift, taking the midpoint of a range like '2.33-2.31'. None on failure."""
    val = str(val).strip()
    m = re.match(r'^([\d.]+)-([\d.]+)$', val)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    try:
        return float(val)
    except ValueError:
        return None


def find_col(header, *candidates):
    """First header name matching a candidate (case-insensitive), else None."""
    hl = [h.lower() for h in header]
    for c in candidates:
        if c.lower() in hl:
            return header[hl.index(c.lower())]
    return None


def _is_exchangeable(label):
    """OH / NH / NH2 / COOH protons — recorded in 1H.csv but excluded from h_nmr,
    matching benchmark.pkl (where h_nmr holds only protons with a 13C partner)."""
    s = str(label).upper().replace(' ', '')
    return 'OH' in s or 'NH' in s or 'COOH' in s or 'CO2H' in s


def read_1h_csv(path):
    """List of float 1H shifts, excluding exchangeable protons."""
    shifts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return shifts
        shift_col = find_col(reader.fieldnames, 'Shift', 'shift', 'h_ppm',
                             '1H_Shift', '1H_shift')
        atom_col = find_col(reader.fieldnames, 'Atom', 'atom', 'label', 'position')
        if shift_col is None:
            return shifts
        for row in reader:
            if atom_col and _is_exchangeable(row.get(atom_col, '')):
                continue
            v = parse_shift(row.get(shift_col, ''))
            if v is not None:
                shifts.append(v)
    return shifts


def read_13c_csv(path):
    """List of float 13C shifts."""
    shifts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return shifts
        shift_col = find_col(reader.fieldnames, 'Shift', 'shift', 'c_ppm',
                             '13C_Shift', '13C_shift')
        if shift_col is None:
            return shifts
        for row in reader:
            v = parse_shift(row.get(shift_col, ''))
            if v is not None:
                shifts.append(v)
    return shifts


def read_hsqc_csv(path):
    """List of [c_ppm, h_ppm, phase]. Phase defaults to +1 when absent."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        h_col = find_col(reader.fieldnames, '1H_Shift', '1H_shift', 'h_ppm',
                         'proton_shift', 'H_Shift')
        c_col = find_col(reader.fieldnames, '13C_Shift', '13C_shift', 'c_ppm',
                         'carbon_shift', 'C_Shift')
        p_col = find_col(reader.fieldnames, 'Phase', 'phase', 'dept', 'DEPT')
        if h_col is None or c_col is None:
            return rows
        for row in reader:
            h = parse_shift(row.get(h_col, ''))
            c = parse_shift(row.get(c_col, ''))
            if h is None or c is None:
                continue
            p = 1.0
            if p_col:
                pv = parse_shift(row.get(p_col, ''))
                if pv is not None:
                    p = float(pv)
            rows.append([c, h, p])
    return rows
