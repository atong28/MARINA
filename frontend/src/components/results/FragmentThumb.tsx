import { memo, useEffect, useRef, useState } from 'react'
import { fragmentSvg } from '../../services/fragmentThumbs'

interface FragmentThumbProps {
  fragmentSmiles: string
  atomSymbol: string
  alt: string
}

/**
 * One bit-panel thumbnail, drawn in the browser when it nears the viewport. A full
 * panel lists ~13k fragments, so drawing them all up front would stall the page.
 * Loaded through <img> like every other depiction, so the markup stays inert.
 */
function FragmentThumb({ fragmentSmiles, atomSymbol, alt }: FragmentThumbProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const [visible, setVisible] = useState(false)
  const [svg, setSvg] = useState<string | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el || visible) return
    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true)
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true)
          observer.disconnect()
        }
      },
      { rootMargin: '400px 0px' },    // start a screenful early
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [visible])

  useEffect(() => {
    if (!visible) return
    let cancelled = false
    fragmentSvg(fragmentSmiles, atomSymbol)
      .then((s) => { if (!cancelled) setSvg(s) })
      .catch(() => { /* leave the empty placeholder */ })
    return () => { cancelled = true }
  }, [visible, fragmentSmiles, atomSymbol])

  if (!svg) return <span ref={ref} className="bit-panel__thumb bit-panel__thumb--empty" />
  return (
    <img
      className="bit-panel__thumb"
      src={`data:image/svg+xml,${encodeURIComponent(svg)}`}
      alt={alt}
    />
  )
}

export default memo(FragmentThumb)
