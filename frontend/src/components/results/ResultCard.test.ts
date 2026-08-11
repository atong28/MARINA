/**
 * Which depiction a card shows for a given toggle state.
 *
 * The two images are not interchangeable: `svg` is the similarity map, so the
 * "off" state must never fall back to it.
 */
import { describe, it, expect } from 'vitest'
import { pickDepiction } from './ResultCard'

const BOTH = { svg: 'highlighted', plain_svg: 'plain' }

describe('pickDepiction', () => {
  it('shows the similarity map when highlighting is on', () => {
    expect(pickDepiction(BOTH, true)).toBe('highlighted')
  })

  it('shows the plain drawing when highlighting is off', () => {
    expect(pickDepiction(BOTH, false)).toBe('plain')
  })

  it('falls back to the plain drawing when no map was rendered', () => {
    expect(pickDepiction({ plain_svg: 'plain' }, true)).toBe('plain')
  })

  it('never falls back to the map when highlighting is off', () => {
    expect(pickDepiction({ svg: 'highlighted' }, false)).toBeUndefined()
  })

  it('returns undefined when the server rendered nothing', () => {
    expect(pickDepiction({}, true)).toBeUndefined()
    expect(pickDepiction({}, false)).toBeUndefined()
  })
})
