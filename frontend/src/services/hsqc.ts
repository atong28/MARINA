/**
 * HSQC rows from the spreadsheet's NaN-padded flat [H, C, I, ...] array.
 *
 * The third column is the multiplicity-edit sign (+ CH/CH3, − CH2). It is either
 * given for every peak or for none: an HSQC with the whole column blank is a
 * shifts-only spectrum, which the server answers with its shifts-only HSQC model;
 * a column filled for some peaks only is incomplete data.
 */

export type HsqcRow = [number, number, number | null]

export interface HsqcRows {
  rows: HsqcRow[]
  /** Every row has shifts but none has a sign/intensity: a shifts-only HSQC. */
  unsigned: boolean
}

/** Rows that have both shifts; a row counts as signed when its third value is finite. */
export function hsqcRows(flat: number[]): HsqcRows {
  const rows: HsqcRow[] = []
  for (let i = 0; i + 2 < flat.length; i += 3) {
    const [h, c, v] = [flat[i], flat[i + 1], flat[i + 2]]
    if (Number.isFinite(h) && Number.isFinite(c)) rows.push([h, c, Number.isFinite(v) ? v : null])
  }
  const unsigned = rows.length > 0 && rows.every((r) => r[2] === null)
  return { rows: unsigned ? rows : rows.filter((r) => r[2] !== null), unsigned }
}

/** The /predict `hsqc` field: [H, C, I, ...], with every I null for a shifts-only HSQC. */
export function hsqcPayload(flat: number[]): Array<number | null> {
  return hsqcRows(flat).rows.flat()
}

/**
 * Spreadsheet rows to flag as incomplete. Each row is [H, C, I] with null for a
 * blank cell. Anything partly filled is incomplete, except H + C without I when
 * no row has I (a shifts-only HSQC).
 */
export function incompleteHsqcRows(rows: Array<[number | null, number | null, number | null]>): number {
  const anySign = rows.some(([, , i]) => i !== null)
  return rows.filter(([h, c, i]) => {
    const filled = [h, c, i].filter((v) => v !== null).length
    if (filled === 0 || filled === 3) return false
    return anySign || h === null || c === null
  }).length
}
