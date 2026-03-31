<script lang="ts">
  interface Props {
    variant?: 'divider' | 'header' | 'footer';
  }
  let { variant = 'divider' }: Props = $props();
</script>

{#if variant === 'divider'}
  <div class="ornament ornament-divider" aria-hidden="true">
    <span class="ornament-rule"></span>
    <span class="ornament-glyph">❧ ✦ ❧</span>
    <span class="ornament-rule"></span>
  </div>
{:else if variant === 'header'}
  <div class="ornament ornament-header" aria-hidden="true">⸻ ✦ ⸻</div>
{:else}
  <div class="ornament ornament-footer" aria-hidden="true">~ ❧ ~</div>
{/if}

<style>
  .ornament {
    text-align: center;
    color: var(--gold);
    font-size: 1rem;
    letter-spacing: 0.25em;
    margin: 1.5rem 0;
    user-select: none;
  }

  .ornament-divider {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .ornament-rule {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--gold), transparent);
  }

  .ornament-glyph {
    flex-shrink: 0;
    color: var(--gold);
  }

  .ornament-header,
  .ornament-footer {
    letter-spacing: 0.4em;
  }
</style>
