export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Region = { id: string; name: string; box: Box };
export type RegisteredObject = {
  id: string;
  name: string;
  label: string;
  registered_at_ms: number;
  box: Box;
  reference_url: string;
};
export type Run = {
  id: string;
  video_id: string;
  backend: string;
  status: 'queued' | 'running' | 'cancelling' | 'cancelled' | 'complete' | 'failed';
  processed_ms: number;
  observation_count: number;
  elapsed_seconds: number;
  created_at: string;
  error: string | null;
  config: Record<string, unknown>;
  provenance: Record<string, unknown>;
  object_ids: string[];
  regions: Region[];
};
export type Recording = {
  id: string;
  title: string;
  original_name: string;
  duration_ms: number;
  width: number;
  height: number;
  fps: number;
  sha256: string;
  is_fixture: boolean;
  /** 'recording' while a camera capture is still growing; 'sealed' once hashed. */
  live_status: 'recording' | 'sealed' | null;
  capture: {
    source: string;
    fps: number;
    frames: number;
    dropped: number;
    /** Wall-clock time of the capture's 0 ms, recorded when its first frame arrived. */
    first_frame_at?: string | null;
  } | null;
  media_url: string;
  poster_url: string;
  regions: Region[];
  objects: RegisteredObject[];
  runs: Run[];
};
export type Evidence = {
  observation_id: string;
  video_id: string;
  run_id: string;
  at_ms: number;
  box: Box;
  zone: string | null;
  /** How this frame related to the registration camera pose. */
  scene_reference: 'registered' | 'compensated' | 'unavailable';
  /** Registration pixels mapped into this frame, present only on a compensated observation. */
  scene_transform: number[][] | null;
  frame_url: string;
  media_url: string;
};
export type MemoryState = {
  object_id: string;
  name: string;
  status: 'visible' | 'last_seen' | 'unknown' | 'not_observed';
  as_of_ms: number;
  last_observed_ms: number | null;
  zone: string | null;
  current_zone: string | null;
  reason: string | null;
  evidence: Evidence | null;
};
export type MemoryEvent = {
  id: string;
  object_id: string;
  name: string;
  at_ms: number;
  kind: string;
  zone: string | null;
  previous_zone: string | null;
  reason: string | null;
  observation_id: string;
};
export type Answer = {
  id: string;
  run_id: string;
  question: string;
  answer: string;
  as_of_ms: number;
  intent: string;
  states: MemoryState[];
  events: MemoryEvent[];
  evidence: Evidence[];
  skills: { name: string; sha256: string; purpose: string }[];
  tools: Record<string, unknown>[];
  planner: 'local' | 'agent' | 'stepfun';
  planner_model: string | null;
  warnings: string[];
  elapsed_seconds: number;
};
export type Capabilities = {
  version: string;
  planner: boolean;
  planner_model: string | null;
  cosmos: boolean;
  cosmos_backend: 'http' | 'transformers';
  cosmos_model: string | null;
  cosmos_reference_adapter_model?: string | null;
  cosmos_base_model?: string | null;
  cosmos_thinking: boolean;
  identity: boolean;
  sam2: boolean;
  sam2_model: string | null;
  rtdetr_installed: boolean;
  discovery: { rtdetr: boolean; cosmos: boolean };
  max_upload_mb: number;
  max_video_seconds: number;
};
export type Proposal = {
  suggested_name: string;
  label: string;
  box: Box;
  score: number | null;
};
export type Suggestions = {
  at_ms: number;
  backend: 'rtdetr' | 'cosmos';
  proposals: Proposal[];
  provenance: Record<string, unknown>;
  elapsed_seconds: number;
  limits: string;
};
export type FrameReport = {
  id: number;
  at_ms: number;
  present: boolean;
  x: number | null;
  y: number | null;
  zone: string | null;
  zone_name: string | null;
  /** How this reviewed frame related to the registration camera pose, when it was checked. */
  camera?: 'compensated' | 'unavailable' | null;
  note: string;
};
export type VisualReview = {
  id: string;
  question: string;
  mode?: 'freeform' | 'grounded';
  target?: string | null;
  frame_reports?: FrameReport[];
  summary: string;
  uncertainty: string;
  notice: string;
  as_of_ms: number;
  model: string;
  elapsed_seconds: number;
  authoritative: false;
  evidence_frame_ids: number[];
  frames: { id: number; at_ms: number; sha256: string; frame_url: string }[];
  skills: { name: string; sha256: string; purpose: string }[];
  provenance: {
    backend: string;
    device: string;
    attempts: number;
    prompt_sha256: string;
    thinking?: boolean;
    served_model?: { id: string; parent: string | null; adapter: boolean };
    served_presence_model?: { id: string; parent: string | null; adapter: boolean };
    presence_veto_model?: string;
    fusion_rule?: string;
  };
};
/** "Cosmos-Reason2-8B + findback-grounding-v1" when a LoRA adapter is served on a base model. */
export function reviewerLabel(
  model: string | null | undefined,
  base: string | null | undefined,
): string {
  const short = (name: string) => name.split('/').pop() || name;
  const name = short(model || 'Cosmos');
  return base ? `${short(base)} + ${name}` : name;
}
export function plannerLabel(model: string | null | undefined): string {
  if (!model) return 'Local query workflow';
  const short = model.split('/').pop() || model;
  return /nemotron/i.test(short) ? `Nemotron agent · ${short}` : `Tool agent · ${short}`;
}
export function time(ms: number | null | undefined): string {
  if (ms == null) return '—';
  return `${Math.floor(ms / 60000)
    .toString()
    .padStart(2, '0')}:${((ms / 1000) % 60).toFixed(1).padStart(4, '0')}`;
}
export function statusLabel(status: string): string {
  return (
    (
      {
        visible: 'Visible at cutoff',
        last_seen: 'Last seen',
        unknown: 'Uncertain',
        not_observed: 'Not yet observed',
      } as Record<string, string>
    )[status] || status
  );
}
