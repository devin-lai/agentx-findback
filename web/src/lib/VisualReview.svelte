<script lang="ts">
  import { time, type VisualReview } from './types';
  let { review, onreplay }: { review: VisualReview; onreplay: (at: number) => void } = $props();
  const reports = $derived(new Map((review.frame_reports || []).map((r) => [r.id, r])));
  const model = $derived(review.model.split('/').pop() || review.model);
</script>

<article class="review-card">
  <span class="eyebrow"
    >{review.mode === 'grounded' ? 'GROUNDED' : 'FREE-FORM'} VISUAL REVIEW · {model} · {time(
      review.as_of_ms,
    )}</span
  >
  {#if review.mode === 'grounded'}<p class="muted small">
      Target: {review.target} · pink marks show the model's proposed match. Inspect the source; a point
      does not verify identity.
    </p>{/if}
  <p class="muted small">{review.question}</p>
  <p><strong>Model interpretation:</strong> {review.summary}</p>
  <p class="muted small">{review.uncertainty}</p>
  <small>{review.notice}</small>
  <div class="review-frames" aria-label="Frames supplied to Cosmos">
    {#each review.frames as frame}
      {@const report = reports.get(frame.id)}
      <button
        type="button"
        class:cited={review.evidence_frame_ids.includes(frame.id)}
        aria-label={`Replay review frame ${frame.id} at ${time(frame.at_ms)}`}
        onclick={() => onreplay(frame.at_ms)}
      >
        <div class="review-image">
          <img
            src={frame.frame_url}
            alt={`Source frame ${frame.id} at ${time(frame.at_ms)}`}
            loading="lazy"
          />
          {#if report?.present && report.x != null && report.y != null}
            <span
              class="review-point"
              role="img"
              aria-label={`Model point in frame ${frame.id}`}
              style:left={`${report.x * 100}%`}
              style:top={`${report.y * 100}%`}
            ></span>
          {/if}
        </div>
        <span
          >#{frame.id} · {time(frame.at_ms)}{review.evidence_frame_ids.includes(frame.id)
            ? ' · cited'
            : ''}</span
        >
        {#if report}<span
            class="frame-report"
            class:present={report?.present}
            title={report.camera === 'compensated'
              ? 'The camera had moved; the model point was read in the registration view before naming the area.'
              : report.camera === 'unavailable'
                ? 'The camera had moved and could not be related to the registration view, so no area is named.'
                : undefined}
            >{report.present
              ? `Model: ${report.zone_name ?? 'unassigned area'}${report.camera === 'compensated' ? ' ·  camera moved' : ''}`
              : 'No model match'}</span
          >{/if}
      </button>
    {/each}
  </div>
  <details class="review-provenance">
    <summary>Model and evidence provenance · {review.elapsed_seconds.toFixed(1)}s</summary>
    <p>
      {review.model}{review.provenance.served_model?.adapter
        ? ` · LoRA adapter on ${review.provenance.served_model.parent}`
        : ''}{review.provenance.presence_veto_model
        ? ` · presence checked by ${review.provenance.presence_veto_model}`
        : ''} · {review.provenance.backend} · {review.provenance.device}{review.provenance.thinking
        ? ' · reasoning enabled'
        : ''}
    </p>
    <p>
      {review.provenance.attempts} generation attempt(s). Frame hashes verify supplied pixels; they do
      not verify the model's claims.
    </p>
    {#each review.skills as skill}
      <p>{skill.name}<br /><code>{skill.sha256}</code></p>
    {:else}<p>Review Skills disabled for this request.</p>{/each}
  </details>
</article>
