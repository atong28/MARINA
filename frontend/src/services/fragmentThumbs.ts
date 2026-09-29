/**
 * Page-side access to the fragment-drawing worker: one worker per page, started
 * on the first thumbnail, with every drawing cached for the page's lifetime.
 */
import type { DrawReply, DrawRequest } from './fragmentWorker'

let worker: Worker | null = null
let nextId = 0
const pending = new Map<number, (svg: string | null) => void>()
// Vocabulary-bounded: at most one entry per fragment the model knows.
const drawn = new Map<string, Promise<string | null>>()

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./fragmentWorker.ts', import.meta.url), { type: 'module' })
    worker.addEventListener('message', (event: MessageEvent<DrawReply>) => {
      const { id, svg, error } = event.data
      if (error) console.warn('fragment drawing failed:', error)
      pending.get(id)?.(svg)
      pending.delete(id)
    })
  }
  return worker
}

/** SVG markup for a bit's fragment, or null if it cannot be drawn. */
export function fragmentSvg(fragmentSmiles: string, atomSymbol = ''): Promise<string | null> {
  const key = `${fragmentSmiles}\u0000${atomSymbol}`
  let svg = drawn.get(key)
  if (!svg) {
    svg = new Promise<string | null>((resolve) => {
      const id = nextId++
      pending.set(id, resolve)
      const request: DrawRequest = { id, fragmentSmiles, atomSymbol }
      getWorker().postMessage(request)
    })
    drawn.set(key, svg)
  }
  return svg
}
