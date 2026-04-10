import { HealthResponse } from '../../services/api'
import './StatusIndicator.css'

interface StatusIndicatorProps {
  health?: HealthResponse
}

function StatusIndicator({ health }: StatusIndicatorProps) {
  const ready = health?.model_loaded ?? false
  return (
    <div className="status-indicator">
      <span className={`status-indicator__dot ${ready ? 'ready' : 'loading'}`} />
      <span className="status-indicator__text">
        {health === undefined ? 'Connecting…' : ready ? 'Ready' : 'Loading model…'}
      </span>
    </div>
  )
}

export default StatusIndicator
