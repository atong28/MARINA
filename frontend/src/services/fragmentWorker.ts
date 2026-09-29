/**
 * Dedicated worker that draws bit-panel fragment thumbnails with RDKit.js.
 *
 * RDKit.js's Emscripten bindings compile functions with `new Function()`, which
 * the page's Content-Security-Policy forbids. A worker loaded from a URL takes its
 * policy from its own response, and the edge nginx serves this file with one that
 * allows eval (nginx/security-headers-worker.conf) — so eval is permitted here,
 * where the only input is vocabulary fragment strings, and never in the page.
 */
import initRDKit, { type MainModule } from '@rdkit/rdkit'
import { drawFragment } from './fragmentDrawing'

// The package's exports map does not list the wasm, so it is addressed by path.
import rdkitWasmUrl from '../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm?url'

export interface DrawRequest { id: number; fragmentSmiles: string; atomSymbol: string }
export interface DrawReply { id: number; svg: string | null; error?: string }

let rdkit: Promise<MainModule> | null = null

self.addEventListener('message', async (event: MessageEvent<DrawRequest>) => {
  const { id, fragmentSmiles, atomSymbol } = event.data
  let reply: DrawReply
  try {
    rdkit ??= initRDKit({ locateFile: () => rdkitWasmUrl })
    reply = { id, svg: drawFragment(await rdkit, fragmentSmiles, atomSymbol) }
  } catch (err) {
    rdkit = null
    reply = { id, svg: null, error: String(err) }
  }
  self.postMessage(reply)
})
