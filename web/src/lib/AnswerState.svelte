<script lang="ts">
  import { AlertTriangle, Check, Clock, EyeOff, Play } from '@lucide/svelte';
  import { time, type Answer, type MemoryState } from './types';

  let {
    answer,
    oncontext,
  }: {
    answer: Answer;
    oncontext?: (answer: Answer) => void;
  } = $props();

  function heading(state: MemoryState): string {
    if (state.status === 'visible') return 'Supported in the recording';
    if (state.status === 'unknown')
      return state.reason === 'identity_ambiguous' || state.reason === 'identity_unconfirmed'
        ? 'Identity uncertain at this time'
        : 'Current position uncertain';
    if (state.status === 'last_seen') return 'No supported current position';
    return 'No confirmed sighting yet';
  }
</script>

{#each answer.states as state (state.object_id)}
  <div class={`answer-state ${state.status}`} aria-label={`State of ${state.name} at cutoff`}>
    <div class="answer-state-top">
      <span class="answer-state-icon" aria-hidden="true">
        {#if state.status === 'visible'}<Check
            size={15}
          />{:else if state.status === 'unknown'}<AlertTriangle
            size={15}
          />{:else if state.status === 'last_seen'}<Clock size={15} />{:else}<EyeOff
            size={15}
          />{/if}
      </span>
      <div>
        <strong>{heading(state)}</strong>
        <span>{state.name} · cutoff {time(answer.as_of_ms)}</span>
      </div>
    </div>
    {#if state.status !== 'visible' && state.evidence}
      <div class="answer-state-history">
        <span>Last confirmed in source · {time(state.evidence.at_ms)}</span>
        {#if oncontext}<button type="button" onclick={() => oncontext?.(answer)}>
            <Play size={12} /> Inspect cutoff frame
          </button>{/if}
      </div>
    {/if}
  </div>
{/each}
