/**
 * Shows where a running prediction sits in the server's queue.
 *
 * The /predict POST blocks until the result is ready, so it cannot report on
 * itself. Instead the client mints a request id, sends it with the POST, and
 * polls GET /api/queue/{id} alongside it.
 */
import { useQueuePosition } from '../../services/api'
import './QueueStatus.css'

interface QueueStatusProps {
  /** The in-flight request's id, or null when nothing is running. */
  requestId: string | null
}

function QueueStatus({ requestId }: QueueStatusProps) {
  const { data, isError } = useQueuePosition(requestId)

  // Before the first poll lands — or if the status endpoint is unreachable —
  // say nothing rather than guessing; the spinner already conveys "working".
  if (!requestId || isError || !data) return null

  if (data.state === 'queued') {
    const waitingBehind = data.position
    return (
      <div className="queue-status queue-status--waiting">
        <span className="queue-status__badge">#{waitingBehind}</span>
        <span>
          Waiting in queue — {waitingBehind === 1 ? 'next up' : `${waitingBehind} ahead of you`}
          {data.workers > 0 && ` · ${data.workers} worker${data.workers === 1 ? '' : 's'} busy`}
        </span>
      </div>
    )
  }

  return (
    <div className="queue-status">
      <span className="queue-status__badge queue-status__badge--running">▶</span>
      <span>
        Running now
        {data.queued > 0 && ` · ${data.queued} other${data.queued === 1 ? '' : 's'} waiting`}
      </span>
    </div>
  )
}

export default QueueStatus
