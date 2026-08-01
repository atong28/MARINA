/**
 * Discover and load bundled spectral data examples.
 * New examples can be added simply by dropping JSON files into src/data/examples/.
 */

export interface ExampleData {
  name: string
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  mw: number | null
}

export interface ExampleMeta {
  /** Display name (from the JSON "name" field, or the filename stem as fallback) */
  name: string
  /** Filename stem (without .json) used as a stable key */
  stem: string
}

// Vite discovers all example JSON files at build time.
const exampleModules = import.meta.glob<{ default: ExampleData }>(
  '../data/examples/*.json',
  { eager: false },
)

/**
 * Return a sorted list of available examples.
 *
 * Each file is fetched to read its display name. They load in parallel rather
 * than one after another, and a file that fails to load falls back to its stem
 * instead of aborting the whole list.
 */
export async function getAvailableExamples(): Promise<ExampleMeta[]> {
  const stems = Object.keys(exampleModules)
    .map((path) => ({ path, stem: path.split('/').pop()?.replace(/\.json$/, '') ?? '' }))
    .filter(({ stem }) => stem && stem !== 'template')

  const entries = await Promise.all(
    stems.map(async ({ path, stem }) => {
      try {
        const mod = await exampleModules[path]()
        const data: ExampleData = (mod.default ?? mod) as ExampleData
        return { name: data.name || stem, stem }
      } catch {
        return { name: stem, stem }
      }
    }),
  )

  return entries.sort((a, b) => a.name.localeCompare(b.name))
}

/**
 * Load the full spectral data for a given filename stem.
 * @param stem  The key returned in ExampleMeta.stem
 */
export async function loadExample(stem: string): Promise<ExampleData> {
  const path = `../data/examples/${stem}.json`
  const loader = exampleModules[path]
  if (!loader) throw new Error(`Example not found: ${stem}`)
  const mod = await loader()
  return (mod.default ?? mod) as ExampleData
}
