/**
 * Substructure thumbnails for the bit panel, drawn in the browser with RDKit.js.
 *
 * The fragments come from the model's fixed vocabulary (~13k of them), so the
 * server used to draw every one per request: ~13 s of CPU and ~90 MB of SVG for a
 * full panel. They are drawn in a Web Worker (fragmentWorker.ts, driven by
 * fragmentThumbs.ts), each only once it scrolls into view (FragmentThumb.tsx).
 *
 * The fragments are *partial* structures cut out of a molecule (Chem.PathToSubmol):
 * an aromatic environment clipped out of its ring arrives as "cc(C)oc(c)c", whose
 * lowercase atoms are non-ring and fail a normal sanitize. They are therefore
 * parsed unsanitised, and every atom is written as a bracket atom with no
 * hydrogens: the open valences are bonds to the rest of the molecule, and left
 * alone RDKit fills them with hydrogens, drawing a lone oxygen as "H2O".
 */
import type { MainModule } from '@rdkit/rdkit'

export const THUMB_SIZE = 120

// A bracket atom, a two-letter organic-subset atom, a one-letter one, or anything
// else (bonds, ring closures, branches) one character at a time.
const TOKEN = /\[[^\]]*\]|Cl|Br|[BCNOPSFIbcnops*]|./g
const ORGANIC = /^(?:Cl|Br|[BCNOPSFIbcnops*])$/
// [isotope] symbol [chirality] [H count] [rest: charge, atom class]
const BRACKET = /^\[(\d*)([A-Z][a-z]?|[a-z][a-z]?|\*)(@[A-Z0-9@]*)?(H\d*)?(.*)\]$/

/** Rewrite a SMILES so no atom carries implicit or explicit hydrogens. */
export function withoutHydrogens(smiles: string): string {
  return smiles.replace(TOKEN, (token) => {
    if (token.startsWith('[')) {
      const m = BRACKET.exec(token)
      return m ? `[${m[1]}${m[2]}${m[3] ?? ''}${m[5]}]` : token
    }
    return ORGANIC.test(token) ? `[${token}]` : token
  })
}

const PARSE = JSON.stringify({ sanitize: false, kekulize: false, removeHs: false })
const DRAW = JSON.stringify({
  width: THUMB_SIZE,
  height: THUMB_SIZE,
  kekulize: false,
  prepareMolsBeforeDrawing: false,
  clearBackground: false,
})

/**
 * Draw one vocabulary fragment as SVG markup. Radius-0 bits carry no fragment
 * SMILES — they are one atom — so the centre-atom symbol is drawn instead.
 */
export function drawFragment(
  rdkit: MainModule, fragmentSmiles: string, atomSymbol = '',
): string | null {
  const smiles = fragmentSmiles || atomSymbol
  if (!smiles) return null
  const mol = rdkit.get_mol(withoutHydrogens(smiles), PARSE)
  if (!mol) return null
  try {
    return mol.get_svg_with_highlights(DRAW)
  } catch {
    return null
  } finally {
    mol.delete()
  }
}
