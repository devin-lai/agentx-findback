<script lang="ts">
  import { onMount } from 'svelte';
  import {
    ArrowRight,
    ArrowUpRight,
    Upload,
    Plus,
    Search,
    Play,
    Film,
    Layers,
    Clock,
    Scan,
    Download,
    LoaderCircle,
    X,
    Trash2,
    ChevronRight,
    Sparkles,
    LockKeyhole,
    Database,
    Check,
    Video,
    Move,
    Camera,
    Square,
  } from '@lucide/svelte';
  import { api, ApiError, downloadEvidence, errorMessage } from './lib/api';
  import {
    time,
    statusLabel,
    type Recording,
    type RegisteredObject,
    type Run,
    type MemoryState,
    type MemoryEvent,
    type Answer,
    type Evidence,
    type Capabilities,
    type VisualReview,
    plannerLabel,
    reviewerLabel,
  } from './lib/types';
  import Registration from './lib/Registration.svelte';
  import EvidenceCard from './lib/EvidenceCard.svelte';
  import AnswerState from './lib/AnswerState.svelte';
  import Trace from './lib/Trace.svelte';
  import VisualReviewCard from './lib/VisualReview.svelte';

  let recordings = $state<Recording[]>([]);
  let recording = $state<Recording | null>(null);
  let run = $state<Run | null>(null);
  let states = $state<MemoryState[]>([]);
  let events = $state<MemoryEvent[]>([]);
  let answers = $state<Answer[]>([]);
  let showEarlierAnswers = $state(false);
  let capabilities = $state<Capabilities | null>(null);
  let error = $state('');
  let busy = $state('');
  let question = $state('');
  let asking = $state(false);
  let exportingAnswer = $state('');
  let exportError = $state('');
  let useAgent = $state(true);
  const agentLabel = $derived(
    /nemotron/i.test(capabilities?.planner_model || '') ? 'Nemotron agent' : 'Tool agent',
  );
  let backend = $state('reference');
  let selectedObject = $state('');
  let cutoff = $state(0);
  let playingAt = $state(0);
  let replayEnd = $state<number | null>(null);
  let registering = $state(false);
  let token = $state('');
  let needsLogin = $state(false);
  let tab = $state<'memory' | 'history'>('memory');
  let visualReviews = $state<VisualReview[]>([]);
  let reviewBusy = $state(false);
  let answerRunVersion = 0;
  let player = $state<HTMLVideoElement>();
  let uploader: HTMLInputElement;
  let regionNames = $state<string[]>([]);
  let editingRegions = $state(false);
  // Live capture from this browser's camera. Frames are timestamped by the server on arrival.
  let liveStream = $state<MediaStream | null>(null);
  let liveVideoId = $state('');
  // Frames are read from a detached element, so capture continues while another recording is
  // open and does not depend on how (or whether) the preview is laid out.
  let liveView: HTMLVideoElement | null = null;
  let liveSent = $state(0);
  let liveStopping = $state(false);
  let liveTimer: ReturnType<typeof setInterval> | undefined;
  let liveInFlight = false;
  const liveCanvas = typeof document === 'undefined' ? null : document.createElement('canvas');
  const LIVE_FPS = 5;
  // A hung request must not stop polling or frame uploads until the browser gives up.
  const POLL_TIMEOUT_MS = 15_000;
  const isLive = $derived(recording?.live_status === 'recording');
  const capturingHere = $derived(isLive && !!liveStream && liveVideoId === recording?.id);
  const working = $derived(
    run?.status === 'queued' || run?.status === 'running' || run?.status === 'cancelling',
  );
  // A memory that follows a live capture can be asked while it is still being built.
  const following = $derived(isLive && run?.status === 'running');
  const ready = $derived(run?.status === 'complete' || following);
  const progress = $derived(
    recording && run && recording.duration_ms > 0
      ? Math.min(100, Math.round((run.processed_ms / recording.duration_ms) * 100))
      : 0,
  );
  const liveLag = $derived(
    following && recording && run ? Math.max(0, recording.duration_ms - run.processed_ms) : 0,
  );
  const activeCount = $derived(states.filter((s) => s.status === 'visible').length);
  const activeAnswers = $derived(answers.slice().reverse());
  const backendAvailable = $derived(
    backend === 'reference' ||
      (backend === 'rtdetr' && !!capabilities?.rtdetr_installed) ||
      (backend === 'cosmos' && !!capabilities?.cosmos) ||
      (backend === 'sam2' && !!capabilities?.sam2),
  );
  const runObjects = $derived(
    (recording?.objects || []).filter((object) => !run || run.object_ids.includes(object.id)),
  );

  async function refresh() {
    try {
      const [caps, items] = await Promise.all([
        api<Capabilities>('/v1/capabilities'),
        api<Recording[]>('/v1/videos'),
      ]);
      capabilities = caps;
      recordings = items;
      needsLogin = false;
      if (!recording && items.length) await selectRecording(items[0]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) needsLogin = true;
      else error = errorMessage(e);
    }
  }
  async function selectRecording(item: Recording) {
    recording = item;
    cutoff = item.duration_ms;
    playingAt = 0;
    regionNames = item.regions.map((r) => r.name);
    await selectRun(item.runs[0] || null);
  }
  async function selectRun(nextRun: Run | null) {
    answerRunVersion += 1;
    asking = false;
    run = nextRun;
    backend = nextRun?.backend || 'reference';
    replayEnd = null;
    player?.pause();
    states = [];
    events = [];
    answers = [];
    showEarlierAnswers = false;
    selectedObject = '';
    question = '';
    visualReviews = [];
    error = '';
    exportError = '';
    if (
      nextRun?.status === 'complete' ||
      (nextRun?.status === 'running' && recording?.live_status === 'recording')
    ) {
      const id = nextRun.id;
      try {
        const [history, savedReviews] = await Promise.all([
          api<Answer[]>(`/v1/runs/${id}/questions`),
          api<VisualReview[]>(`/v1/runs/${id}/reviews`),
          readMemory(),
        ]);
        if (run?.id === id) {
          answers = history;
          visualReviews = savedReviews;
        }
      } catch (e) {
        if (run?.id === id) error = errorMessage(e);
      }
    }
  }
  function chooseRun(event: Event) {
    const id = (event.target as HTMLSelectElement).value;
    const chosen = recording?.runs.find((candidate) => candidate.id === id);
    if (chosen) void selectRun(chosen);
  }
  function backendLabel(value: string) {
    return (
      { reference: 'Reference tracking', rtdetr: 'RT-DETR', cosmos: 'Cosmos', sam2: 'SAM 2.1' }[
        value
      ] || value
    );
  }
  async function readMemory() {
    if (!run || !ready) return;
    if (following) cutoff = run.processed_ms;
    const id = run.id,
      t = cutoff;
    const [nextStates, nextEvents] = await Promise.all([
      api<MemoryState[]>(`/v1/runs/${id}/state?at_ms=${t}`, {
        signal: AbortSignal.timeout(POLL_TIMEOUT_MS),
      }),
      api<MemoryEvent[]>(`/v1/runs/${id}/events?at_ms=${t}`, {
        signal: AbortSignal.timeout(POLL_TIMEOUT_MS),
      }),
    ]);
    if (run?.id === id && cutoff === t) {
      states = nextStates;
      events = nextEvents;
      if (following) noteLiveEvents(id, nextEvents);
    }
  }
  // Memory events that arrive while a live capture is followed, newest first.
  let liveFeed = $state<MemoryEvent[]>([]);
  let liveFeedRun = '';
  let seenEvents = new Set<string>();
  function noteLiveEvents(runId: string, next: MemoryEvent[]) {
    if (liveFeedRun !== runId) {
      liveFeedRun = runId;
      liveFeed = [];
      seenEvents = new Set();
    }
    const fresh = next.filter((event) => !seenEvents.has(event.id));
    fresh.forEach((event) => seenEvents.add(event.id));
    if (fresh.length) liveFeed = [...fresh.reverse(), ...liveFeed].slice(0, 5);
  }
  async function exportAnswer(id: string) {
    const runId = run?.id;
    exportingAnswer = id;
    exportError = '';
    try {
      await downloadEvidence(id);
    } catch (e) {
      if (run?.id === runId) exportError = errorMessage(e);
    } finally {
      exportingAnswer = '';
    }
  }
  async function poll() {
    if (!run || !working) return;
    const id = run.id;
    try {
      const updated = await api<Run>(`/v1/runs/${id}`, {
        signal: AbortSignal.timeout(POLL_TIMEOUT_MS),
      });
      if (run?.id !== id) return;
      run = updated;
      if (['complete', 'failed', 'cancelled'].includes(updated.status) && recording) {
        const updatedVideo = await api<Recording>(`/v1/videos/${recording.id}`);
        if (recording?.id !== updatedVideo.id) return;
        recording = updatedVideo;
        recordings = recordings.map((r) => (r.id === updatedVideo.id ? updatedVideo : r));
        if (updated.status === 'complete') {
          cutoff = recording.duration_ms;
          await readMemory();
        }
      }
    } catch (e) {
      error = errorMessage(e);
    }
  }
  async function pollLive() {
    if (!recording || recording.live_status !== 'recording') return;
    const id = recording.id;
    try {
      const updated = await api<Recording>(`/v1/videos/${id}`, {
        signal: AbortSignal.timeout(POLL_TIMEOUT_MS),
      });
      if (recording?.id !== id) return;
      recording = updated;
      recordings = recordings.map((r) => (r.id === id ? updated : r));
      if (run) run = updated.runs.find((candidate) => candidate.id === run?.id) || run;
      if (updated.live_status !== 'recording' && liveVideoId === id) releaseCamera();
      if (following) await readMemory();
    } catch (e) {
      error = errorMessage(e);
    }
  }
  onMount(() => {
    void refresh();
    let polling = false;
    const interval = setInterval(() => {
      if (!polling) {
        polling = true;
        void Promise.all([poll(), pollLive()]).finally(() => (polling = false));
      }
    }, 1000);
    return () => {
      clearInterval(interval);
      releaseCamera();
    };
  });
  function attachStream(node: HTMLVideoElement, stream: MediaStream | null) {
    node.srcObject = stream;
    return {
      update(next: MediaStream | null) {
        node.srcObject = next;
      },
    };
  }
  async function startLive() {
    if (!navigator.mediaDevices?.getUserMedia) {
      error = 'This browser cannot open a camera on this page. Use localhost or HTTPS.';
      return;
    }
    busy = 'live';
    error = '';
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const item = await api<Recording>('/v1/live', {
        method: 'POST',
        body: JSON.stringify({ title: `Live camera · ${stamp}`, fps: LIVE_FPS }),
      });
      const view = document.createElement('video');
      view.muted = true;
      view.playsInline = true;
      view.srcObject = stream;
      await view.play();
      liveView = view;
      liveStream = stream;
      liveVideoId = item.id;
      liveSent = 0;
      recordings = [item, ...recordings];
      await selectRecording(item);
      liveTimer = setInterval(() => void sendLiveFrame(), 1000 / LIVE_FPS);
    } catch (e) {
      stream?.getTracks().forEach((track) => track.stop());
      error =
        e instanceof DOMException ? `The camera is unavailable: ${e.message}` : errorMessage(e);
    } finally {
      busy = '';
    }
  }
  async function sendLiveFrame() {
    const view = liveView;
    if (liveInFlight || !liveStream || !view || view.readyState < 2 || !liveCanvas) return;
    const scale = Math.min(1, 1280 / Math.max(view.videoWidth, view.videoHeight));
    liveCanvas.width = Math.round(view.videoWidth * scale);
    liveCanvas.height = Math.round(view.videoHeight * scale);
    liveCanvas.getContext('2d')?.drawImage(view, 0, 0, liveCanvas.width, liveCanvas.height);
    const id = liveVideoId;
    liveInFlight = true;
    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        liveCanvas.toBlob(resolve, 'image/jpeg', 0.85),
      );
      if (!blob) return;
      const response = await fetch(`/api/v1/live/${id}/frames`, {
        method: 'POST',
        body: blob,
        headers: { 'Content-Type': 'image/jpeg' },
        credentials: 'same-origin',
        signal: AbortSignal.timeout(POLL_TIMEOUT_MS),
      });
      // A late reply about a capture that was already stopped must not end a newer one.
      if (liveVideoId !== id) return;
      if (response.status === 409 || response.status === 404) releaseCamera();
      else if (response.ok) liveSent += 1;
    } catch {
      // A dropped request loses one frame; the next tick sends a fresh one.
    } finally {
      liveInFlight = false;
    }
  }
  function releaseCamera() {
    if (liveTimer) clearInterval(liveTimer);
    liveTimer = undefined;
    liveStream?.getTracks().forEach((track) => track.stop());
    liveStream = null;
    if (liveView) liveView.srcObject = null;
    liveView = null;
  }
  async function stopLive() {
    if (!recording || liveStopping) return;
    const id = recording.id;
    liveStopping = true;
    if (liveVideoId === id) releaseCamera();
    try {
      const sealed = await api<Recording>(`/v1/live/${id}/stop`, { method: 'POST' });
      recordings = recordings.map((r) => (r.id === id ? sealed : r));
      if (recording?.id === id) {
        recording = sealed;
        if (run) run = sealed.runs.find((candidate) => candidate.id === run?.id) || run;
      }
    } catch (e) {
      error = errorMessage(e);
    } finally {
      liveStopping = false;
    }
  }
  async function login() {
    busy = 'login';
    try {
      await api('/auth/login', { method: 'POST', body: JSON.stringify({ token }) });
      token = '';
      await refresh();
    } catch (e) {
      error = errorMessage(e);
    } finally {
      busy = '';
    }
  }
  async function upload(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    busy = 'upload';
    error = '';
    try {
      const data = new FormData();
      data.append('file', file);
      const item = await api<Recording>('/v1/videos', { method: 'POST', body: data });
      recordings = [item, ...recordings];
      await selectRecording(item);
    } catch (e) {
      error = errorMessage(e);
    } finally {
      busy = '';
      uploader.value = '';
    }
  }
  async function loadDemo(scene: 'fixed' | 'bumped' = 'fixed') {
    busy = scene === 'bumped' ? 'demo-bumped' : 'demo';
    error = '';
    try {
      const item = await api<Recording>(`/v1/demo?scene=${scene}`, { method: 'POST' });
      recordings = [item, ...recordings];
      await selectRecording(item);
      await buildMemory('reference');
    } catch (e) {
      error = errorMessage(e);
    } finally {
      busy = '';
    }
  }
  async function buildMemory(chosen = backend) {
    if (!recording) return;
    const requestVideo = recording.id;
    error = '';
    try {
      const nextRun = await api<Run>(`/v1/videos/${recording.id}/runs`, {
        method: 'POST',
        body: JSON.stringify({
          backend: chosen,
          sample_fps: chosen === 'cosmos' ? 1 : 5,
          match_threshold: 0.82,
        }),
      });
      if (recording?.id !== requestVideo) return;
      recording = { ...recording, runs: [nextRun, ...recording.runs] };
      recordings = recordings.map((item) => (item.id === requestVideo ? recording! : item));
      await selectRun(nextRun);
    } catch (e) {
      error = errorMessage(e);
    }
  }
  async function savedObject() {
    registering = false;
    if (!recording) return;
    const item = await api<Recording>(`/v1/videos/${recording.id}`);
    if (recording?.id === item.id) recording = item;
    recordings = recordings.map((r) => (r.id === item.id ? item : r));
  }
  async function stopIndexing() {
    if (!run || run.status === 'cancelling') return;
    const id = run.id;
    try {
      const stopped = await api<Run>(`/v1/runs/${id}/cancel`, { method: 'POST' });
      if (run?.id !== id) return;
      run = stopped;
      if (recording) {
        recording = {
          ...recording,
          runs: recording.runs.map((item) => (item.id === id ? stopped : item)),
        };
        const current = recording;
        recordings = recordings.map((item) => (item.id === current.id ? current : item));
      }
    } catch (e) {
      if (run?.id === id) error = errorMessage(e);
    }
  }
  async function changeCutoff() {
    replayEnd = null;
    try {
      await readMemory();
      if (player) player.currentTime = Math.min(cutoff, (recording?.duration_ms || 1) - 1) / 1000;
    } catch (e) {
      error = errorMessage(e);
    }
  }
  async function ask(text = question, objectId?: string, atMs = cutoff) {
    if (!run || !text.trim()) return;
    // While following a live capture, "now" is the newest moment memory has processed.
    if (following && atMs === cutoff) atMs = run.processed_ms;
    if (objectId !== undefined) selectedObject = objectId;
    const requestRun = run.id;
    const requestVersion = answerRunVersion;
    asking = true;
    error = '';
    try {
      const result = await api<Answer>(`/v1/runs/${run.id}/questions`, {
        method: 'POST',
        body: JSON.stringify({
          use_provider: useAgent,
          text,
          at_ms: atMs,
          object_id: selectedObject || null,
        }),
      });
      if (run?.id === requestRun && answerRunVersion === requestVersion) {
        answers = [...answers, result];
        showEarlierAnswers = false;
        question = '';
        if (result.tools.some((tool) => tool.name === 'review_frames')) {
          try {
            const savedReviews = await api<VisualReview[]>(`/v1/runs/${requestRun}/reviews`);
            if (run?.id === requestRun && answerRunVersion === requestVersion)
              visualReviews = savedReviews;
          } catch (e) {
            if (run?.id === requestRun && answerRunVersion === requestVersion)
              error = `Answer saved, but its visual review could not be loaded: ${errorMessage(e)}`;
          }
        }
      }
    } catch (e) {
      if (run?.id === requestRun && answerRunVersion === requestVersion) error = errorMessage(e);
    } finally {
      if (run?.id === requestRun && answerRunVersion === requestVersion) asking = false;
    }
  }
  function matchingCandidates(answer: Answer): RegisteredObject[] {
    if (answer.states.length || answer.evidence.length || answer.run_id !== run?.id) return [];
    const selection = [...answer.tools]
      .reverse()
      .find((tool) => Array.isArray(tool.matching_objects));
    if (
      !selection ||
      !Array.isArray(selection.matching_objects) ||
      selection.matching_objects.length < 2
    )
      return [];
    const ids = new Set(
      selection.matching_objects.map((value: unknown) => {
        if (typeof value === 'object' && value !== null && 'object_id' in value)
          return value.object_id;
        return null;
      }),
    );
    return runObjects.filter((obj) => ids.has(obj.id));
  }
  function replay(e: Evidence) {
    if (!player) return;
    const answerCutoff =
      answers.find((a) => a.evidence.some((x) => x.observation_id === e.observation_id))
        ?.as_of_ms ?? cutoff;
    replayEnd = Math.min(answerCutoff, e.at_ms + 1000) / 1000;
    player.currentTime = Math.max(0, e.at_ms / 1000 - 1);
    void player.play().catch(() => {});
  }
  function inspectCutoff(answer: Answer) {
    if (!player || !recording || answer.run_id !== run?.id) return;
    replayEnd = null;
    player.pause();
    player.currentTime = Math.max(0, Math.min(answer.as_of_ms, recording.duration_ms - 1) / 1000);
    player.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  function updatePlayback() {
    if (!player) return;
    playingAt = Math.round(player.currentTime * 1000);
    if (replayEnd !== null && player.currentTime >= replayEnd) {
      player.pause();
      replayEnd = null;
    }
  }
  async function removeVideo() {
    if (!recording || !confirm('Delete this recording, its memory and all evidence?')) return;
    try {
      const id = recording.id;
      if (liveVideoId === id) releaseCamera();
      await api(`/v1/videos/${id}`, { method: 'DELETE' });
      recording = null;
      run = null;
      states = [];
      answers = [];
      visualReviews = [];
      events = [];
      recordings = recordings.filter((v) => v.id !== id);
      if (recordings.length) await selectRecording(recordings[0]);
    } catch (e) {
      error = errorMessage(e);
    }
  }
  async function saveRegions() {
    if (!recording) return;
    try {
      const regions = recording.regions.map((r, i) => ({ ...r, name: regionNames[i] }));
      await api(`/v1/videos/${recording.id}/regions`, {
        method: 'PUT',
        body: JSON.stringify(regions),
      });
      recording = { ...recording, regions };
      editingRegions = false;
    } catch (e) {
      error = errorMessage(e);
    }
  }
  function reviewTarget(): string | null {
    // One registered object named in the question selects the grounded per-frame mode.
    const lower = question.toLowerCase();
    const named = runObjects.filter((o) => lower.includes(o.name.toLowerCase()));
    return named.length === 1 ? named[0].id : null;
  }
  async function visualReview() {
    if (!run || reviewBusy) return;
    error = '';
    const requestRun = run.id;
    reviewBusy = true;
    try {
      const result = await api<VisualReview>(`/v1/runs/${run.id}/review`, {
        method: 'POST',
        body: JSON.stringify({
          text: question || 'Describe only the supported object changes in these frames.',
          object_id: selectedObject || reviewTarget() || null,
          at_ms: cutoff,
        }),
      });
      if (run?.id === requestRun) visualReviews = [...visualReviews, result];
    } catch (e) {
      error = errorMessage(e);
    } finally {
      reviewBusy = false;
    }
  }
  // A live capture's offsets also have a wall-clock time: "last seen at 14:02:31".
  function wall(ms: number | null | undefined) {
    const start = recording?.capture?.first_frame_at;
    if (!start || ms === null || ms === undefined) return '';
    const at = new Date(Date.parse(start) + ms);
    return Number.isNaN(at.getTime())
      ? ''
      : ` · ${at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  }
  function describeEvent(event: MemoryEvent) {
    return event.kind === 'moved'
      ? `${zoneName(event.previous_zone)} → ${zoneName(event.zone)}`
      : event.kind === 'lost'
        ? 'Visual contact lost'
        : event.kind === 'ambiguous'
          ? 'Scene or identity uncertain'
          : `${event.kind === 'reappeared' ? 'Reappeared' : 'First observed'} in ${zoneName(event.zone)}`;
  }
  function zoneName(id: string | null) {
    return (
      (run?.regions || recording?.regions)?.find((r) => r.id === id)?.name || 'Unassigned area'
    );
  }
</script>

<svelte:head
  ><title>AgentX FindBack · Visual memory</title><meta
    name="description"
    content="Trace an object's story through video. Search timestamped observations and replay original evidence."
  /></svelte:head
>
<input
  class="visually-hidden"
  type="file"
  accept="video/mp4,video/quicktime,video/webm,.mkv,.avi"
  bind:this={uploader}
  onchange={upload}
  aria-label="Upload video file"
/>
<div class="app-shell">
  <aside class="sidebar">
    <a class="brand" href="/" aria-label="AgentX FindBack home"
      ><span class="brand-mark"><span></span><span></span><span></span><span></span></span><span
        >Agent<span class="brand-x">X</span><small>FINDBACK</small></span
      ></a
    >
    <div class="workspace-label">WORKSPACE <span>01</span></div>
    <button class="nav-item active"
      ><Layers size={17} /> Visual memory <span class="nav-count">{recordings.length}</span></button
    >
    <div class="sidebar-section">
      <span>RECORDINGS</span><button
        class="icon-button"
        onclick={() => uploader?.click()}
        disabled={!!busy || needsLogin}
        aria-label="Add recording"><Plus size={16} /></button
      >
    </div>
    <div class="recording-list">
      {#each recordings as item}<button
          class:selected={recording?.id === item.id}
          class="recording-item"
          onclick={() => selectRecording(item).catch((e) => (error = errorMessage(e)))}
          ><span class="recording-icon"><Film size={16} /></span><span class="recording-name"
            >{item.title}<small
              >{time(item.duration_ms)} · {item.live_status === 'recording'
                ? 'Live · recording'
                : item.live_status === 'sealed'
                  ? 'Live capture'
                  : item.is_fixture
                    ? 'Controlled sample'
                    : 'Uploaded video'}</small
            ></span
          >{#if recording?.id === item.id}<span class="dot"></span>{/if}</button
        >{/each}{#if !recordings.length}<p class="sidebar-empty">
          Your recordings will appear here.
        </p>{/if}
    </div>
    <div class="sidebar-bottom">
      <div class="mini-orbit"><Scan size={22} /></div>
      <strong>Memory with receipts.</strong>
      <p>Every location has a time.<br />Every answer has evidence.</p>
      <div class="sidebar-footer">
        <span>AGENTX / 2026</span><span>v{capabilities?.version || '0.1.0'}</span>
      </div>
    </div>
  </aside>
  <div class="main-shell">
    <header class="topbar">
      <div class="breadcrumbs">Workspace <ChevronRight size={13} /><span>Visual memory</span></div>
      <a class="mobile-brand" href="/" aria-label="AgentX FindBack home"
        ><span class="brand-mark mobile-mark" aria-hidden="true"
          ><span></span><span></span><span></span><span></span></span
        ><strong>AgentX</strong><span>FindBack</span></a
      >
      <div class="topbar-right">
        <span class="quiet-badge"
          ><span class="dot"></span>{capabilities?.planner
            ? plannerLabel(capabilities.planner_model)
            : 'Local query workflow'}</span
        ><span class="avatar">AX</span>
      </div>
    </header>
    <main>
      {#if error}<div class="error-banner global-error" role="alert">
          {error}<button class="icon-button" onclick={() => (error = '')} aria-label="Dismiss error"
            ><X size={16} /></button
          >
        </div>{/if}
      {#if needsLogin}
        <section class="login-panel">
          <LockKeyhole size={28} />
          <h1>Your memory, protected.</h1>
          <p>Enter the deployment access token to open this workspace.</p>
          <form
            onsubmit={(e) => {
              e.preventDefault();
              void login();
            }}
          >
            <label
              >Access token<input
                type="password"
                bind:value={token}
                autocomplete="current-password"
              /></label
            ><button class="primary" disabled={!!busy}
              >Open workspace <ArrowRight size={16} /></button
            >
          </form>
        </section>
      {:else}
        <section class="page-heading">
          <div>
            <div class="eyebrow">
              <span class="small-line"></span> SEE IT. REMEMBER IT. FIND IT.
            </div>
            <h1>Give your footage a memory.</h1>
            <p>Find what moved. Know when. See the evidence.</p>
          </div>
          <div class="heading-actions">
            <button
              class="secondary"
              onclick={startLive}
              disabled={!!busy || !!liveStream || liveStopping}
              title="Record from this device's camera; memory follows while it records"
              >{#if busy === 'live'}<LoaderCircle size={17} class="spin" />{:else}<Camera
                  size={17}
                />{/if} Live camera</button
            ><button class="primary" onclick={() => uploader?.click()} disabled={!!busy}
              >{#if busy === 'upload'}<LoaderCircle size={17} class="spin" /> Processing upload…{:else}<Upload
                  size={17}
                /> Upload recording{/if}</button
            >
          </div>
        </section>
        {#if recordings.length > 0}
          <label class="mobile-recordings"
            >Recording
            <select
              value={recording?.id}
              onchange={(e) => {
                const item = recordings.find((v) => v.id === e.currentTarget.value);
                if (item) void selectRecording(item).catch((e) => (error = errorMessage(e)));
              }}
            >
              {#each recordings as item}<option value={item.id}>{item.title}</option>{/each}
            </select>
          </label>
        {/if}
        {#if !recording}
          <section class="empty-workspace">
            <div class="empty-visual">
              <div class="scan-corner tl"></div>
              <div class="scan-corner tr"></div>
              <div class="scan-corner bl"></div>
              <div class="scan-corner br"></div>
              <div class="orbit-ring"></div>
              <div class="orbit-ring inner"></div>
              <div class="empty-icon"><Scan size={42} strokeWidth={1.2} /></div>
              <span class="floating-label"
                ><span class="dot"></span> Waiting for a first observation</span
              >
            </div>
            <span class="eyebrow">YOUR FIRST VISUAL MEMORY</span>
            <h2>A place for everything.<br />A memory of where it went.</h2>
            <p>
              Upload a fixed-camera recording, register the objects you care about, and turn their
              movements into a searchable timeline.
            </p>
            <div class="empty-actions">
              <button class="primary" onclick={() => uploader?.click()} disabled={!!busy}
                ><Upload size={16} /> Upload a video</button
              ><button
                class="secondary"
                onclick={startLive}
                disabled={!!busy || !!liveStream || liveStopping}
                >{#if busy === 'live'}<LoaderCircle size={16} class="spin" />{:else}<Camera
                    size={15}
                  />{/if} Live camera</button
              ><button class="secondary" onclick={() => loadDemo('fixed')} disabled={!!busy}
                >{#if busy === 'demo'}<LoaderCircle size={16} class="spin" />{:else}<Play
                    size={15}
                  />{/if} Try controlled sample</button
              ><button class="secondary" onclick={() => loadDemo('bumped')} disabled={!!busy}
                >{#if busy === 'demo-bumped'}<LoaderCircle size={16} class="spin" />{:else}<Move
                    size={15}
                  />{/if} Try bumped camera</button
              >
            </div>
            <span class="small muted"
              >The bumped sample knocks the camera at six seconds and leaves it there. Nothing on
              the desk moves afterwards, so every region answer must stay what it was.</span
            >
            <span class="small muted"
              >MP4, MOV, WebM, MKV or AVI · up to {capabilities?.max_upload_mb || 256} MB</span
            >
          </section>
          <div class="principles">
            <article>
              <span>01</span>
              <div>
                <h3>Remember changes</h3>
                <p>Follow objects through observations, movement and loss of visibility.</p>
              </div>
            </article>
            <article>
              <span>02</span>
              <div>
                <h3>Ask across time</h3>
                <p>Query the latest indexed view or travel back to a specific moment.</p>
              </div>
            </article>
            <article>
              <span>03</span>
              <div>
                <h3>Return to evidence</h3>
                <p>Every supported location links back to the original recording.</p>
              </div>
            </article>
          </div>
        {:else}
          <div class="workspace-grid">
            <section class="footage-column">
              <div class="panel footage-panel">
                <div class="panel-header">
                  <div>
                    <span class="eyebrow">SOURCE RECORDING</span>
                    <h2>{recording.title}</h2>
                  </div>
                  <div class="header-actions">
                    {#if isLive}<span class="tag live"><span class="dot"></span> Live</span
                      >{:else if recording.live_status === 'sealed'}<span class="tag"
                        >Live capture · sealed</span
                      >{:else if recording.is_fixture}<span class="tag amber"
                        >Controlled sample</span
                      >{:else}<span class="tag">Uploaded</span>{/if}<button
                      class="icon-button"
                      onclick={removeVideo}
                      disabled={working}
                      aria-label="Delete recording"><Trash2 size={16} /></button
                    >
                  </div>
                </div>
                <div class="player-wrapper">
                  {#if isLive}
                    {#if capturingHere}<video
                        class="live-preview"
                        use:attachStream={liveStream}
                        muted
                        playsinline
                        autoplay
                        aria-label="Live camera preview"
                      ></video>{:else if recording.duration_ms > 0}<img
                        class="live-preview"
                        src={`/api/v1/videos/${recording.id}/frame?at_ms=${recording.duration_ms - 1}`}
                        alt="Latest recorded live frame"
                      />{:else}<div class="live-waiting">
                        <LoaderCircle size={20} class="spin" /> Waiting for the first frame…
                      </div>{/if}<span class="recording-watermark live"
                      ><span></span> LIVE · {time(recording.duration_ms)}</span
                    >
                  {:else}
                    <video
                      bind:this={player}
                      src={recording.media_url}
                      poster={recording.poster_url}
                      controls
                      playsinline
                      preload="metadata"
                      ontimeupdate={updatePlayback}><track kind="captions" /></video
                    ><span class="recording-watermark"><span></span> RECORDED VIDEO</span>
                  {/if}
                </div>
                <div class="source-meta">
                  <span><Film size={13} />{recording.width} × {recording.height}</span><span
                    ><Clock size={13} />{time(recording.duration_ms)}</span
                  ><span class="mono"
                    >{isLive ? 'Hashed when stopped' : `SHA ${recording.sha256.slice(0, 10)}`}</span
                  >
                </div>
                {#if isLive}<div class="live-controls">
                    <span
                      >{capturingHere
                        ? `Recording from this camera · ${liveSent} frames sent`
                        : `Recording from ${recording.capture?.source === 'push' ? 'a pushed stream' : 'another device'} · ${recording.capture?.frames ?? 0} frames`}</span
                    ><button class="secondary" onclick={stopLive} disabled={liveStopping}
                      >{#if liveStopping}<LoaderCircle size={14} class="spin" /> Sealing…{:else}<Square
                          size={13}
                        /> Stop &amp; seal{/if}</button
                    >
                  </div>
                  <p class="live-note">
                    Register objects on the newest frame, then build memory: it follows the camera
                    and answers about <em>now</em> while recording. Stopping hashes the file and makes
                    it an ordinary, exportable recording.
                  </p>{/if}
                {#if recording.is_fixture}<div class="fixture-note">
                    Generated software fixture for validating the workflow. It is not a real-world
                    model benchmark.
                  </div>{/if}
              </div>
              <div class="panel registration-panel">
                <div class="panel-header compact">
                  <h3>Objects to remember <span class="count">{recording.objects.length}</span></h3>
                  <button
                    class="text-button"
                    onclick={() => {
                      player?.pause();
                      // A live capture registers on its newest readable frame.
                      if (isLive && recording) playingAt = Math.max(0, recording.duration_ms - 1);
                      registering = true;
                    }}
                    disabled={working || (isLive && !recording.duration_ms)}
                    ><Plus size={15} /> Register object</button
                  >
                </div>
                {#if !recording.objects.length}<div class="inline-empty">
                    <Scan size={22} />
                    <div>
                      <strong>Start with a visual reference</strong>
                      <p>Pause at a clear frame and draw a box around your first object.</p>
                    </div>
                  </div>{:else}<div class="object-chips">
                    {#each recording.objects as object}<div class="object-chip">
                        <img src={object.reference_url} alt={object.name} />
                        <div>
                          <strong>{object.name}</strong><small
                            >{object.label} · from {time(object.registered_at_ms)}</small
                          >
                        </div>
                      </div>{/each}
                  </div>{/if}
                <div class="regions-row">
                  <span>Search regions</span>{#each recording.regions as region}<span
                      class="region-chip">{region.name}</span
                    >{/each}<button
                    class="text-button"
                    onclick={() => (editingRegions = !editingRegions)}
                    disabled={working}>Rename</button
                  >
                </div>
                {#if editingRegions}<div class="region-editor">
                    {#each regionNames as name, i}<input
                        aria-label={`Region ${i + 1} name`}
                        bind:value={regionNames[i]}
                        maxlength="60"
                      />{/each}<button class="secondary" onclick={saveRegions}
                      ><Check size={15} /> Save</button
                    >
                  </div>{/if}
                <div class="index-controls">
                  <label class="backend-picker"
                    >Perception<select bind:value={backend}
                      ><option value="reference">Reference tracking · OpenCV</option><option
                        value="rtdetr"
                        disabled={!capabilities?.rtdetr_installed}
                        >RT-DETR + ByteTrack{capabilities?.rtdetr_installed
                          ? capabilities?.identity
                            ? ' + DINOv2 identity'
                            : ''
                          : ' · install vision extra'}</option
                      ><option value="cosmos" disabled={!capabilities?.cosmos}
                        >{(capabilities?.cosmos_model || 'Cosmos').split('/').pop()} grounding · any object{capabilities?.cosmos
                          ? ' · 1 fps'
                          : ' · configure Cosmos'}</option
                      ><option value="sam2" disabled={!capabilities?.sam2}
                        >SAM 2.1 temporal tracking{capabilities?.sam2
                          ? ' · preview'
                          : ' · configure checkpoint'}</option
                      ></select
                    ></label
                  ><button
                    class="primary"
                    onclick={() => buildMemory()}
                    disabled={working || !recording.objects.length || !backendAvailable}
                    >{#if working}<LoaderCircle size={16} class="spin" /> Indexing…{:else}<Layers
                        size={16}
                      />{run ? 'Rebuild memory' : 'Build memory'}{/if}</button
                  >
                </div>
                {#if !backendAvailable}<p class="backend-unavailable" role="status">
                    {backendLabel(backend)} is unavailable on this device. Saved memory and evidence still
                    work; select an available backend to build a new version.
                  </p>{/if}
                {#if recording.runs.length > 1}<label class="run-picker">
                    Memory version
                    <select
                      aria-label="Memory version"
                      value={run?.id}
                      onchange={chooseRun}
                      disabled={working}
                    >
                      {#each recording.runs as version, i}
                        <option value={version.id}>
                          {`Version ${recording.runs.length - i} · ${backendLabel(version.backend)} · ${version.status}`}
                        </option>
                      {/each}
                    </select>
                    <span>Each version keeps its own observations, answers and evidence.</span>
                  </label>{/if}
                {#if working}<div class="job-progress">
                    <span style={`width:${progress}%`}></span>
                  </div>
                  <div class="progress-meta">
                    <span
                      >{run?.status === 'queued'
                        ? 'Queued'
                        : run?.status === 'cancelling'
                          ? 'Finishing the current frame…'
                          : following
                            ? `Following the live camera · ${(liveLag / 1000).toFixed(1)} s behind`
                            : `Reading observations · ${progress}%`}</span
                    ><span>{time(run?.processed_ms)} processed</span>
                  </div>
                  <button
                    class="secondary job-stop"
                    onclick={stopIndexing}
                    disabled={run?.status === 'cancelling'}
                  >
                    <X size={14} />
                    {run?.status === 'cancelling' ? 'Stopping…' : 'Stop indexing'}
                  </button>{/if}
                {#if run?.status === 'cancelled'}<p class="job-stopped" role="status">
                    Indexing stopped. Build a new memory version to continue.
                  </p>{/if}
                {#if run?.status === 'failed'}<div class="error-banner" role="alert">
                    {run.error}
                  </div>{/if}
                {#if ready}<div class="index-receipt">
                    <Check size={14} />{run?.observation_count} observations{following
                      ? ' so far'
                      : ` · ${run?.elapsed_seconds.toFixed(1)}s indexing`} · {run?.backend ===
                    'reference'
                      ? 'Classical reference tracking'
                      : run?.backend === 'cosmos'
                        ? 'Cosmos per-frame grounding'
                        : run?.backend === 'sam2'
                          ? 'SAM 2.1 temporal tracking'
                          : 'Neural object detection'}<a
                      href={`/api/v1/runs/${run?.id}/export`}
                      aria-label="Export run evidence"><Download size={14} /></a
                    >
                  </div>{/if}
              </div>
              <div class="panel memory-panel">
                <div class="panel-header compact">
                  <div class="tab-list">
                    <button class:chosen={tab === 'memory'} onclick={() => (tab = 'memory')}
                      >Object memory</button
                    ><button class:chosen={tab === 'history'} onclick={() => (tab = 'history')}
                      >Event history <span class="count">{events.length}</span></button
                    >
                  </div>
                  <span class="small muted"
                    >{ready ? `${activeCount} visible at cutoff` : 'Awaiting index'}</span
                  >
                </div>
                {#if following}<div class="time-travel live">
                    <div>
                      <span><Clock size={14} /> Live memory</span><strong class="mono"
                        >{time(cutoff)}</strong
                      >
                    </div>
                    <p>
                      Answers use everything processed so far. Stop the recording to travel back in
                      time or export a report.
                    </p>
                  </div>{:else if ready}<div class="time-travel">
                    <div>
                      <span><Clock size={14} /> Query cutoff</span><strong class="mono"
                        >{time(cutoff)}</strong
                      >
                    </div>
                    <input
                      aria-label="Query cutoff"
                      type="range"
                      min="0"
                      max={recording.duration_ms}
                      step="100"
                      bind:value={cutoff}
                      onchange={changeCutoff}
                    />
                    <p>Answers use observations at or before this point in the recording.</p>
                  </div>{/if}
                {#if tab === 'memory'}<div class="memory-list">
                    {#each states as state}<button
                        class="memory-row"
                        onclick={() => {
                          selectedObject = state.object_id;
                          question = `Where is ${state.name}?`;
                        }}
                        ><span class={`state-dot ${state.status}`}></span>
                        <div class="memory-name">
                          <strong>{state.name}</strong><small
                            >{state.last_observed_ms !== null
                              ? state.status === 'visible'
                                ? `${state.current_zone ? zoneName(state.current_zone) : 'Area unconfirmed'} · ${time(state.last_observed_ms)}${wall(state.last_observed_ms)}`
                                : `Last confirmed ${time(state.last_observed_ms)}${wall(state.last_observed_ms)}`
                              : 'No earlier observation'}</small
                          >
                        </div>
                        <span class={`state-label ${state.status}`}
                          >{statusLabel(state.status)}</span
                        ><ArrowUpRight size={14} /></button
                      >{/each}{#if !states.length}<p class="panel-empty">
                        Build memory to see timestamped object states.
                      </p>{/if}
                  </div>
                {:else}<div class="event-list">
                    {#each events as event}<div class="event-row">
                        <span class="event-time mono">{time(event.at_ms)}</span><span
                          class={`event-marker ${event.kind}`}
                        ></span>
                        <div>
                          <strong>{event.name}</strong>
                          <p>{describeEvent(event)}</p>
                        </div>
                      </div>{/each}{#if !events.length}<p class="panel-empty">
                        No events before the selected cutoff.
                      </p>{/if}
                  </div>{/if}
              </div>
            </section>
            <section class="panel assistant-panel">
              <div class="panel-header">
                <div class="assistant-title">
                  <span class="assistant-icon"><Sparkles size={18} /></span>
                  <div>
                    <h2>Ask your memory</h2>
                    <span class="small muted">Grounded in what was observed</span>
                  </div>
                </div>
                <span class="tag">{capabilities?.planner ? agentLabel : 'Local'}</span>
              </div>
              <div class="assistant-body">
                {#if following && liveFeed.length}<div class="live-feed" aria-live="polite">
                    <span class="eyebrow">Live activity</span>
                    {#each liveFeed as event (event.id)}<div class="live-feed-row">
                        <span class={`event-marker ${event.kind}`}></span><strong
                          >{event.name}</strong
                        ><span>{describeEvent(event)}</span><span class="mono"
                          >{time(event.at_ms)}{wall(event.at_ms)}</span
                        >
                      </div>{/each}
                  </div>{/if}
                {#if !answers.length}<div class="assistant-welcome">
                    <span class="assistant-orbit"><Search size={27} strokeWidth={1.5} /></span>
                    <h3>Where did it go?</h3>
                    <p>
                      Ask about an object, trace its history, or find the last moment it was
                      visible.
                    </p>
                    <div class="suggestions">
                      {#each runObjects.slice(0, 2) as obj}<button
                          disabled={!ready}
                          onclick={() => ask(`Where was ${obj.name} last seen?`, obj.id)}
                          ><Clock size={14} /> Last seen: {obj.name}<ArrowUpRight
                            size={14}
                          /></button
                        >{/each}{#if runObjects[0]}<button
                          disabled={!ready}
                          onclick={() =>
                            ask(`Show the history of ${runObjects[0].name}`, runObjects[0].id)}
                          ><Layers size={14} /> Trace its movement<ArrowUpRight size={14} /></button
                        >{/if}
                      {#if capabilities?.planner && useAgent && runObjects.length > 1}
                        <button
                          disabled={!ready}
                          onclick={() => ask('Show the history of the item that disappeared.', '')}
                        >
                          <Sparkles size={14} /> Which item disappeared?<ArrowUpRight size={14} />
                        </button>
                      {/if}
                    </div>
                  </div>{/if}
                {#each activeAnswers as answer, index (answer.id)}
                  {#if index === 1}<button
                      class="earlier-answers-toggle"
                      aria-expanded={showEarlierAnswers}
                      onclick={() => (showEarlierAnswers = !showEarlierAnswers)}
                    >
                      {showEarlierAnswers ? 'Hide' : 'Show'}
                      {activeAnswers.length - 1} earlier
                      {activeAnswers.length === 2 ? 'answer' : 'answers'}
                      <ChevronRight size={15} class={showEarlierAnswers ? 'expanded' : ''} />
                    </button>{/if}
                  {#if index === 0 || showEarlierAnswers}<article class="answer">
                      <div class="question-bubble">
                        <strong>{answer.question}</strong><span
                          >as of {time(answer.as_of_ms)}{wall(answer.as_of_ms)}</span
                        >
                      </div>
                      <div class="answer-label"><span class="brand-dot"></span> FINDBACK</div>
                      <AnswerState {answer} oncontext={isLive ? undefined : inspectCutoff} />
                      <p class="answer-text">{answer.answer}</p>
                      {#if matchingCandidates(answer).length > 1}
                        <div class="clarification" aria-label="Choose a matching object">
                          <span class="small muted"
                            >Choose a match · as of {time(answer.as_of_ms)}</span
                          >
                          <div class="clarification-options">
                            {#each matchingCandidates(answer) as object}
                              <button
                                class="secondary"
                                disabled={asking || !ready}
                                onclick={() => ask(answer.question, object.id, answer.as_of_ms)}
                              >
                                {object.name}<ArrowRight size={13} />
                              </button>
                            {/each}
                          </div>
                        </div>
                      {/if}
                      {#if answer.evidence.length}<div class="evidence-grid">
                          {#each answer.evidence.slice(0, 6) as evidence}<EvidenceCard
                              {evidence}
                              regions={run?.regions || recording?.regions || []}
                              onreplay={replay}
                            />{/each}
                        </div>{/if}{#each answer.warnings as warning}<p class="warning-note">
                          {warning}
                        </p>{/each}<Trace {answer} />
                      <div class="answer-export">
                        {#if following}<span
                            >Stop the live recording to hash it and export a verifiable report.</span
                          >{:else}<button
                            class="text-button"
                            disabled={!!exportingAnswer}
                            onclick={() => exportAnswer(answer.id)}
                          >
                            {#if exportingAnswer === answer.id}<LoaderCircle
                                size={13}
                                class="spin"
                              />{:else}<Download size={13} />{/if}
                            {exportingAnswer === answer.id
                              ? 'Preparing report…'
                              : 'Save evidence report'}
                          </button>
                          <span>Offline report + original frames</span>{/if}
                      </div>
                    </article>{/if}{/each}
                {#if exportError}<p class="warning-note" role="alert">{exportError}</p>{/if}
                {#each visualReviews.slice().reverse() as savedReview (savedReview.id)}
                  <VisualReviewCard
                    review={savedReview}
                    onreplay={(at) => {
                      if (player) {
                        replayEnd = savedReview.as_of_ms / 1000;
                        player.currentTime = at / 1000;
                        void player.play();
                      }
                    }}
                  />
                {/each}
              </div>
              <div class="composer">
                <label class="scope-picker"
                  >Query scope<select bind:value={selectedObject}
                    ><option value="">Resolve from question</option
                    >{#each runObjects as object}<option value={object.id}>{object.name}</option
                      >{/each}</select
                  ></label
                >
                <form
                  onsubmit={(e) => {
                    e.preventDefault();
                    void ask();
                  }}
                >
                  <textarea
                    aria-label="Ask your memory"
                    bind:value={question}
                    placeholder={ready
                      ? 'Where was the object last seen?'
                      : 'Build memory to start asking questions…'}
                    disabled={!ready || asking}
                    rows="2"
                    onkeydown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        void ask();
                      }
                    }}></textarea><button
                    class="send-button"
                    aria-label="Send question"
                    disabled={!ready || asking || !question.trim()}
                    >{#if asking}<LoaderCircle size={17} class="spin" />{:else}<ArrowRight
                        size={18}
                      />{/if}</button
                  >
                </form>
                <div class="composer-footer">
                  <span><LockKeyhole size={11} /> Evidence stays linked to its source</span
                  >{#if capabilities?.planner}<label class="agent-toggle"
                      ><input type="checkbox" bind:checked={useAgent} /> {agentLabel}</label
                    >{/if}{#if capabilities?.cosmos}<button
                      class="text-button"
                      disabled={!ready || reviewBusy || following}
                      title={following
                        ? 'Visual review cites hashed frames: stop the live recording first.'
                        : undefined}
                      onclick={visualReview}
                      >{reviewBusy
                        ? 'Reviewing…'
                        : `Review with ${
                            capabilities?.cosmos_reference_adapter_model &&
                            (selectedObject || reviewTarget())
                              ? reviewerLabel(
                                  capabilities.cosmos_reference_adapter_model,
                                  capabilities.cosmos_model,
                                )
                              : reviewerLabel(
                                  capabilities?.cosmos_model,
                                  capabilities?.cosmos_base_model,
                                )
                          }`}</button
                    >{/if}
                </div>
              </div>
            </section>
          </div>
        {/if}
        <footer class="page-footer">
          <span><Database size={12} /> Persistent visual memory</span><span
            >Observed facts · Explicit uncertainty · Original evidence</span
          >
        </footer>
      {/if}
    </main>
  </div>
</div>
{#if registering && recording}<Registration
    {recording}
    atMs={playingAt}
    {capabilities}
    onclose={() => (registering = false)}
    onsaved={() => void savedObject().catch((e) => (error = errorMessage(e)))}
  />{/if}
