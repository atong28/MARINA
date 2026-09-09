/**
 * Which depiction layers a card shows for a given toggle state.
 *
 * The two images are not interchangeable: `svg` is the similarity-map wash
 * drawn under the line drawing, so the "off" state must never include it and
 * it can never stand in for the drawing.
 */
import { describe, it, expect } from 'vitest'
import { pickDepiction } from './ResultCard'

const BOTH = { svg: 'highlighted', plain_svg: 'plain' }

describe('pickDepiction', () => {
  it('layers the map under the drawing when highlighting is on', () => {
    expect(pickDepiction(BOTH, true)).toEqual({ plain: 'plain', map: 'highlighted' })
  })

  it('shows the plain drawing alone when highlighting is off', () => {
    expect(pickDepiction(BOTH, false)).toEqual({ plain: 'plain', map: undefined })
  })

  it('keeps the drawing when no map was rendered', () => {
    expect(pickDepiction({ plain_svg: 'plain' }, true)).toEqual({ plain: 'plain', map: undefined })
  })

  it('never uses the map when highlighting is off', () => {
    expect(pickDepiction({ svg: 'highlighted' }, false).map).toBeUndefined()
  })

  it('returns no layers when the server rendered nothing', () => {
    expect(pickDepiction({}, true)).toEqual({ plain: undefined, map: undefined })
    expect(pickDepiction({}, false)).toEqual({ plain: undefined, map: undefined })
  })
})
