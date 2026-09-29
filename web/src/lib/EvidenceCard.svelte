<script lang="ts">
  import { ArrowUpRight, Move } from '@lucide/svelte';
  import { time, type Evidence, type Region } from './types';
  let {
    evidence,
    regions = [],
    onreplay,
  } = $props<{ evidence: Evidence; regions?: Region[]; onreplay: (e: Evidence) => void }>();
  // The box always belongs to this frame. The badge says whether the region it was read
  // against had to be recovered from the registration view.
  /** Where the regions the user drew ended up in this frame, once the camera moved.
   *  The transform maps registration pixels into this frame; because the overlay is drawn in
   *  percentages of the same frame, the pixel size cancels and only the ratios matter. */
  const projected = $derived.by(() => {
    const m = evidence.scene_transform;
    if (!m || evidence.scene_reference !== 'compensated') return [];
    const at = (x: number, y: number) => {
      const w = m[2][0] * x + m[2][1] * y + m[2][2];
      if (!w) return null;
      return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w];
    };
    return (regions as Region[])
      .map((region: Region) => {
        const b = region.box;
        const corners = [at(b.x1, b.y1), at(b.x2, b.y1), at(b.x2, b.y2), at(b.x1, b.y2)];
        if (corners.some((c) => c === null)) return null;
        return {
          id: region.id,
          name: region.name,
          points: (corners as number[][]).map(([x, y]) => `${x * 100},${y * 100}`).join(' '),
        };
      })
      .filter((r): r is { id: string; name: string; points: string } => r !== null);
  });
  const camera = $derived(
    evidence.scene_reference === 'compensated'
      ? 'Camera moved · region recovered from the registration view'
      : evidence.scene_reference === 'unavailable'
        ? 'Camera moved · no region could be named for this frame'
        : '',
  );
</script>

<button
  class="evidence-card"
  onclick={() => onreplay(evidence)}
  aria-label={`Replay evidence at ${time(evidence.at_ms)}`}
>
  <!-- The image sets this container's height. A fixed aspect ratio would offset the
       normalized box on portrait or non-16:9 source frames. -->
  <div class="evidence-image">
    <img
      src={evidence.frame_url}
      alt={`Original evidence at ${time(evidence.at_ms)}`}
      loading="lazy"
    /><span
      class="evidence-box"
      style={`left:${evidence.box.x1 * 100}%;top:${evidence.box.y1 * 100}%;width:${(evidence.box.x2 - evidence.box.x1) * 100}%;height:${(evidence.box.y2 - evidence.box.y1) * 100}%`}
    ></span>{#if projected.length}<svg
        class="evidence-regions"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
        >{#each projected as region}<polygon
            points={region.points}
            class:named={region.id === evidence.zone}
          />{/each}</svg
      >{/if}{#if camera}<span class="evidence-camera" title={camera}
        ><Move size={11} /> Camera moved</span
      >{/if}
  </div>
  <span class="evidence-caption"
    ><span><span class="dot"></span> Original frame · {time(evidence.at_ms)}</span><ArrowUpRight
      size={15}
    /></span
  >
</button>

<style>
  .evidence-regions {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .evidence-regions polygon {
    fill: none;
    stroke: #f0b42980;
    stroke-width: 1;
    stroke-dasharray: 3 3;
    vector-effect: non-scaling-stroke;
  }
  .evidence-regions polygon.named {
    stroke: #f0b429;
    stroke-width: 1.5;
    stroke-dasharray: none;
    fill: #f0b42926;
  }
  .evidence-camera {
    position: absolute;
    left: 6px;
    top: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px 2px 5px;
    border-radius: 999px;
    font-size: 8px;
    letter-spacing: 0.02em;
    color: #1c1407;
    background: #f0b429e6;
    box-shadow: 0 1px 4px #00000059;
    pointer-events: none;
  }
</style>
