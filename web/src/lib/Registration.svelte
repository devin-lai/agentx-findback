<script lang="ts">
  import { onMount } from 'svelte';
  import { X, Scan, Check, Sparkles } from '@lucide/svelte';
  import { api, errorMessage } from './api';
  import type { Box, Capabilities, Proposal, Recording, Suggestions } from './types';
  import { time } from './types';
  let { recording, atMs, capabilities, onclose, onsaved } = $props<{
    recording: Recording;
    atMs: number;
    capabilities: Capabilities | null;
    onclose: () => void;
    onsaved: () => void;
  }>();
  let name = $state('');
  let label = $state('custom');
  let box = $state<Box | null>(null);
  let origin = $state<{ x: number; y: number } | null>(null);
  let busy = $state(false);
  let error = $state('');
  let proposals = $state<Proposal[]>([]);
  let suggestion = $state<Suggestions | null>(null);
  let suggesting = $state(false);
  let chosen = $state<number | null>(null);
  const backends = $derived(
    (['rtdetr', 'cosmos'] as const).filter((b) => capabilities?.discovery?.[b]),
  );
  const COMMON_CATEGORIES = [
    'scissors',
    'remote',
    'cup',
    'book',
    'mouse',
    'bottle',
    'cell phone',
    'keyboard',
  ];
  // RT-DETR proposes any COCO category, so the list has to admit one that is not common.
  const categories = $derived(
    COMMON_CATEGORIES.includes(label) || label === 'custom'
      ? COMMON_CATEGORIES
      : [...COMMON_CATEGORIES, label].sort(),
  );
  const frameTime = $derived(Math.max(0, Math.min(atMs, recording.duration_ms - 1)));
  let dialog: HTMLDialogElement;
  onMount(() => dialog.showModal());
  function point(e: PointerEvent) {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    };
  }
  function down(e: PointerEvent) {
    origin = point(e);
    box = null;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }
  function move(e: PointerEvent) {
    if (!origin) return;
    const p = point(e);
    box = {
      x1: Math.min(origin.x, p.x),
      y1: Math.min(origin.y, p.y),
      x2: Math.max(origin.x, p.x),
      y2: Math.max(origin.y, p.y),
    };
  }
  async function suggest(backend: 'rtdetr' | 'cosmos') {
    suggesting = true;
    error = '';
    try {
      const result = await api<Suggestions>(`/v1/videos/${recording.id}/suggestions`, {
        method: 'POST',
        body: JSON.stringify({ at_ms: frameTime, backend }),
      });
      suggestion = result;
      proposals = result.proposals;
      chosen = null;
      if (!result.proposals.length)
        error = 'No objects were proposed in this frame. Draw a box instead.';
    } catch (e) {
      error = errorMessage(e);
    } finally {
      suggesting = false;
    }
  }
  function clearSuggestions() {
    proposals = [];
    suggestion = null;
    chosen = null;
  }
  function accept(index: number) {
    const proposal = proposals[index];
    chosen = index;
    box = { ...proposal.box };
    name = proposal.suggested_name;
    label = proposal.label;
  }
  async function save() {
    if (!box) return;
    busy = true;
    error = '';
    try {
      await api(`/v1/videos/${recording.id}/objects`, {
        method: 'POST',
        body: JSON.stringify({ name, label, at_ms: frameTime, box }),
      });
      onsaved();
    } catch (e) {
      error = errorMessage(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="modal-backdrop" role="presentation">
  <dialog
    bind:this={dialog}
    class="registration modal"
    aria-labelledby="registration-title"
    oncancel={onclose}
  >
    <div class="modal-heading">
      <div>
        <span class="eyebrow">Teach a visual reference</span>
        <h2 id="registration-title">Register an object</h2>
      </div>
      <button class="icon-button" onclick={onclose} aria-label="Close registration"
        ><X size={20} /></button
      >
    </div>
    <p class="muted">
      Draw a tight box around a clearly visible object at {time(frameTime)}, or let a model propose
      candidates and just name the one you want. This frame establishes its first available
      reference.
    </p>
    {#if backends.length}
      <div class="suggest-bar">
        {#each backends as backend (backend)}
          <button class="text-button" disabled={suggesting} onclick={() => suggest(backend)}>
            <Sparkles size={15} />
            {suggesting
              ? 'Looking…'
              : backend === 'rtdetr'
                ? 'Suggest objects (RT-DETR)'
                : 'Suggest objects (Cosmos)'}
          </button>
        {/each}
        {#if suggestion}
          <span class="muted small"
            >{proposals.length} proposed in {suggestion.elapsed_seconds.toFixed(1)}s · suggestions
            are not evidence; confirm each one.</span
          >
          {#if proposals.length}
            <button class="text-button" onclick={clearSuggestions}>Clear suggestions</button>
          {/if}
        {/if}
      </div>
    {/if}
    <div
      class="selection-frame"
      style={`aspect-ratio:${recording.width}/${recording.height}`}
      onpointerdown={down}
      onpointermove={move}
      onpointerup={() => (origin = null)}
      role="application"
      aria-label="Draw an object bounding box"
    >
      <img
        src={`/api/v1/videos/${recording.id}/frame?at_ms=${frameTime}`}
        alt="Registration frame"
        draggable="false"
      />
      {#each proposals as proposal, index (index)}
        <button
          class="proposal-box"
          class:chosen={chosen === index}
          style={`left:${proposal.box.x1 * 100}%;top:${proposal.box.y1 * 100}%;width:${(proposal.box.x2 - proposal.box.x1) * 100}%;height:${(proposal.box.y2 - proposal.box.y1) * 100}%`}
          onpointerdown={(e) => e.stopPropagation()}
          onclick={() => accept(index)}
          aria-label={`Use suggestion ${proposal.suggested_name}`}
        >
          <span class="proposal-label"
            >{proposal.suggested_name}{proposal.score != null
              ? ` · ${Math.round(proposal.score * 100)}%`
              : ''}</span
          >
        </button>
      {/each}
      {#if box}<div
          class="selection-box"
          style={`left:${box.x1 * 100}%;top:${box.y1 * 100}%;width:${(box.x2 - box.x1) * 100}%;height:${(box.y2 - box.y1) * 100}%`}
        ></div>{:else if !proposals.length}<span class="selection-hint"
          ><Scan size={18} /> Drag to select an object</span
        >{/if}
    </div>
    <div class="form-grid">
      <label
        >Object name<input
          bind:value={name}
          placeholder="e.g. Red scissors"
          maxlength="80"
        /></label
      ><label
        >Detection category<select bind:value={label}
          ><option value="custom">Custom / reference tracking</option
          >{#each categories as category (category)}<option>{category}</option>{/each}</select
        ></label
      >
    </div>
    {#if box}<details class="coordinate-editor">
        <summary>Adjust normalized bounds with keyboard</summary>
        <div class="form-grid">
          <label
            >Left<input
              aria-label="Box left"
              type="number"
              min="0"
              max="1"
              step="0.01"
              bind:value={box.x1}
            /></label
          ><label
            >Top<input
              aria-label="Box top"
              type="number"
              min="0"
              max="1"
              step="0.01"
              bind:value={box.y1}
            /></label
          ><label
            >Right<input
              aria-label="Box right"
              type="number"
              min="0"
              max="1"
              step="0.01"
              bind:value={box.x2}
            /></label
          ><label
            >Bottom<input
              aria-label="Box bottom"
              type="number"
              min="0"
              max="1"
              step="0.01"
              bind:value={box.y2}
            /></label
          >
        </div>
      </details>{:else}<button
        class="text-button"
        onclick={() => (box = { x1: 0.2, y1: 0.2, x2: 0.5, y2: 0.5 })}
        >Enter bounding coordinates instead</button
      >{/if}
    {#if error}<p class="error-banner" role="alert">{error}</p>{/if}
    <div class="modal-actions">
      <span class="muted small">One object per category for RT-DETR.</span><button
        class="primary"
        disabled={busy || !name.trim() || !box}
        onclick={save}><Check size={16} />{busy ? 'Saving…' : 'Save reference'}</button
      >
    </div>
  </dialog>
</div>

<style>
  .suggest-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
  }
  .proposal-box {
    position: absolute;
    padding: 0;
    background: color-mix(in srgb, var(--accent, #76b900) 12%, transparent);
    border: 2px dashed var(--accent, #76b900);
    border-radius: 3px;
    cursor: pointer;
  }
  .proposal-box:hover,
  .proposal-box:focus-visible {
    background: color-mix(in srgb, var(--accent, #76b900) 26%, transparent);
  }
  .proposal-box.chosen {
    border-style: solid;
    box-shadow: 0 0 0 2px var(--accent, #76b900);
  }
  .proposal-label {
    position: absolute;
    left: 0;
    bottom: 100%;
    max-width: 22ch;
    overflow: hidden;
    padding: 1px 5px;
    font-size: 0.68rem;
    line-height: 1.5;
    white-space: nowrap;
    text-overflow: ellipsis;
    color: #0b0f0b;
    background: var(--accent, #76b900);
    border-radius: 3px 3px 0 0;
  }
</style>
