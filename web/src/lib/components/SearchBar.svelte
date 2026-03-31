<script lang="ts">
  interface Props {
    disabled?: boolean;
    onSearch?: (query: string) => void;
  }
  let { disabled = false, onSearch }: Props = $props();

  let query = $state('');

  function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed) onSearch?.(trimmed);
  }
</script>

<form class="search-bar" onsubmit={handleSubmit}>
  <input
    type="text"
    class="search-input"
    bind:value={query}
    placeholder="Describe the music you seek&hellip;"
    aria-label="Music search query"
    {disabled}
  />
  <button type="submit" class="search-btn" {disabled}>
    Search
  </button>
</form>

<style>
  .search-bar {
    display: flex;
    gap: 0.5rem;
    max-width: 640px;
    margin: 0 auto;
  }

  .search-input {
    flex: 1;
    padding: 0.65rem 1rem;
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.05rem;
    background: var(--cream);
    color: var(--ink);
    border: 1px solid var(--gold);
    border-radius: 2px;
    outline: none;
    transition: box-shadow 0.2s;
  }

  .search-input:focus {
    box-shadow: 0 0 0 2px rgba(197, 165, 90, 0.4);
  }

  .search-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .search-input::placeholder {
    color: var(--dark-brown);
    opacity: 0.5;
  }

  .search-btn {
    padding: 0.65rem 1.5rem;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.95rem;
    background: var(--burgundy);
    color: var(--cream);
    border: none;
    border-radius: 2px;
    cursor: pointer;
    transition: background 0.2s, opacity 0.2s;
    white-space: nowrap;
  }

  .search-btn:hover:not(:disabled) {
    background: #8b3a44;
  }

  .search-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
