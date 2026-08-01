/**
 * API client helpers.
 */
import { describe, it, expect } from 'vitest'
import { isAbortError } from './api'

describe('isAbortError', () => {
  it('recognises the rejection fetch raises on abort', () => {
    // Regression: starting a new search cancels the in-flight one, and the
    // cancelled mutation's error was surfaced as "The user aborted a request"
    // on top of a perfectly good result.
    const err = new Error('The user aborted a request.')
    err.name = 'AbortError'
    expect(isAbortError(err)).toBe(true)
  })

  it('recognises a DOMException-style abort', () => {
    const err = new DOMException('aborted', 'AbortError')
    expect(isAbortError(err)).toBe(true)
  })

  it('does not swallow genuine failures', () => {
    expect(isAbortError(new Error('HTTP 500'))).toBe(false)
    expect(isAbortError(new TypeError('network down'))).toBe(false)
  })

  it('tolerates non-error values', () => {
    expect(isAbortError(null)).toBe(false)
    expect(isAbortError(undefined)).toBe(false)
    expect(isAbortError('AbortError')).toBe(false)
    expect(isAbortError({ name: 'AbortError' })).toBe(false)
  })
})
