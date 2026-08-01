/**
 * Cumulative queries served and distinct visitors, from GET /api/stats.
 * Renders nothing until the first successful fetch, so a backend without the
 * endpoint simply shows no counter rather than an error.
 */
import { useUsageStats } from '../../services/api'
import './UsageCounter.css'

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `${(n / 1_000).toFixed(0)}k`
  return n.toLocaleString()
}

function formatSince(epochSeconds: number): string {
  if (!epochSeconds) return ''
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function UsageCounter() {
  const { data } = useUsageStats()
  if (!data) return null

  const since = formatSince(data.counting_since)

  return (
    <div className="usage-counter" title={since ? `Counting since ${since}` : undefined}>
      <span className="usage-counter__item">
        <strong>{formatCount(data.queries_total)}</strong> queries
      </span>
      <span className="usage-counter__sep">·</span>
      <span className="usage-counter__item">
        <strong>{formatCount(data.unique_clients)}</strong> visitors
      </span>
      {since && <span className="usage-counter__since">since {since}</span>}
    </div>
  )
}

export default UsageCounter
