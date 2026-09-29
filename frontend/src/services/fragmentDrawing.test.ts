import { beforeAll, describe, expect, it } from 'vitest'
import initRDKit, { type MainModule } from '@rdkit/rdkit'
import { drawFragment, withoutHydrogens } from './fragmentDrawing'

describe('withoutHydrogens', () => {
  it('brackets organic-subset atoms so RDKit adds no implicit hydrogens', () => {
    expect(withoutHydrogens('CC(O)Cl')).toBe('[C][C]([O])[Cl]')
    expect(withoutHydrogens('cc(C)oc(c)c')).toBe('[c][c]([C])[o][c]([c])[c]')
  })

  it('strips explicit hydrogen counts from bracket atoms, keeping charge and isotope', () => {
    expect(withoutHydrogens('cc[nH]c')).toBe('[c][c][n][c]')
    expect(withoutHydrogens('C[NH3+]')).toBe('[C][N+]')
    expect(withoutHydrogens('[13CH2]O')).toBe('[13C][O]')
  })

  it('leaves bonds, ring closures and elements beginning with H alone', () => {
    expect(withoutHydrogens('C1=CC#C1')).toBe('[C]1=[C][C]#[C]1')
    expect(withoutHydrogens('C%12CC%12')).toBe('[C]%12[C][C]%12')
    expect(withoutHydrogens('[Hg]Br')).toBe('[Hg][Br]')
  })
})

describe('drawFragment', () => {
  let rdkit: MainModule
  beforeAll(async () => { rdkit = await initRDKit() })

  // Glyphs RDKit draws for atom labels ("O", "H2O", ...) are paths classed atom-N only.
  const labelPaths = (svg: string | null) => (svg?.match(/<path class='atom-\d+'/g) ?? []).length

  it.each([
    'CCCCC',            // plain chain
    'C=O',              // double bond
    'ccc',              // aromatic atoms clipped out of their ring
    'cc(C)oc(c)c',      // aromatic heterocycle fragment
    'CC(O)C(O)C(C)O',   // several dangling valences
    'cc[nH]c(=O)c',     // explicit ring hydrogen
  ])('draws the partial structure %s', (frag) => {
    expect(drawFragment(rdkit, frag)).toContain('<svg')
  })

  it('falls back to the centre atom for radius-0 bits', () => {
    expect(drawFragment(rdkit, '', 'O')).toContain('<svg')
  })

  it('returns null when there is nothing to draw', () => {
    expect(drawFragment(rdkit, '', '')).toBeNull()
  })

  it('does not invent hydrogens on open valences', () => {
    // A plain parse fills them in: a lone oxygen draws as "H2O", ethers as "OH".
    const naive = (smiles: string) => {
      const mol = rdkit.get_mol(smiles)!
      try { return mol.get_svg() } finally { mol.delete() }
    }
    expect(labelPaths(drawFragment(rdkit, '', 'O'))).toBeLessThan(labelPaths(naive('O')))
    expect(labelPaths(drawFragment(rdkit, 'CC(O)C(O)C(C)O'))).toBeLessThan(labelPaths(naive('CC(O)C(O)C(C)O')))
  })
})
