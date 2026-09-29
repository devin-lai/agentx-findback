<script lang="ts">
  import { Check, ChevronDown } from '@lucide/svelte';
  import { plannerLabel, type Answer } from './types';
  let { answer } = $props<{ answer: Answer }>();
  const label = $derived(
    answer.planner === 'agent'
      ? plannerLabel(answer.planner_model)
      : answer.planner === 'stepfun'
        ? 'StepFun planner (previous release)'
        : 'Local query workflow',
  );
</script>

<details class="trace">
  <summary
    ><span
      ><Check size={14} />
      {answer.evidence.length ? 'Source and time checked' : 'No supported location'}</span
    ><span>{answer.tools.length} steps <ChevronDown size={14} /></span></summary
  >
  <div class="trace-content">
    <p class="muted small">
      {label} · {answer.elapsed_seconds.toFixed(2)}s · facts constrained to the selected cutoff
    </p>
    {#each answer.tools as tool, i}<div class="trace-row">
        <span class="trace-index">{i + 1}</span><code>{String(tool.name)}</code><span
          class="muted small"
          >{typeof tool.result === 'string'
            ? tool.result
            : tool.cutoff_ms != null
              ? `${tool.cutoff_ms} ms`
              : ''}</span
        >
      </div>
      {#if typeof tool.thought === 'string' && tool.thought}<p class="trace-thought">
          {tool.thought}
        </p>{/if}
      {#if typeof tool.interpretation === 'string' && tool.interpretation}<p class="trace-thought">
          Criteria checked: {tool.interpretation}
        </p>{/if}{/each}
    <div class="skill-hashes">
      {#each answer.skills as skill}<div>
          <span>{skill.name}</span><code>{skill.sha256.slice(0, 12)}</code>
        </div>{/each}
    </div>
  </div>
</details>
