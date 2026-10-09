import { parseErrorCode } from './errorReport'

/**
 * True only for the gateway's own `instances_disabled` 403 from
 * `/api/instances`: the one denial a UI treats as "feature off" and leaves
 * silent. That endpoint's other 403s (a non-owner caller, a Slack-origin
 * request) return false, so its two callers, the top bar and the sidebar's
 * embedded-pane notice, show them as failures. `InstancesPanel` and
 * `RemoteCrewPanel` keep their own message-based checks for now.
 *
 * Checked by shape (`status` + `body`) rather than `instanceof ApiError`, so it
 * reads the same field the client's own benign-denial parser keys on.
 */
export function isInstancesDisabledError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false
  const { status, body } = error as { status?: unknown; body?: unknown }
  return status === 403 && typeof body === 'string' && parseErrorCode(body) === 'instances_disabled'
}
