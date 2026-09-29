export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData))
    headers.set('Content-Type', 'application/json');
  const response = await fetch(`/api${path}`, { ...options, headers, credentials: 'same-origin' });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* Preserve the HTTP failure when no JSON is available. */
    }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}
export function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === 'TimeoutError')
    return 'The server did not respond in time. It will be retried.';
  return error instanceof Error ? error.message : 'An unexpected error occurred.';
}

export async function downloadEvidence(answerId: string): Promise<void> {
  const response = await fetch(`/api/v1/questions/${encodeURIComponent(answerId)}/bundle`, {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      typeof body.detail === 'string' ? body.detail : `Export failed (${response.status}).`,
    );
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = `findback-evidence-${answerId}.zip`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Give the browser time to start the download before releasing its backing data.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
