# factkit — semantic recall for Hermes holographic memory

A standalone Hermes plugin that runs alongside the built-in `holographic` memory
provider. It uses the same database — no fork, no migration, no changes to the
provider — and adds:

- a semantic-recall block injected before each model call (`pre_llm_call`),
  merged with the provider's own keyword/HRR prefetch;
- history-aware dedupe: a fact already shown earlier in the conversation is not
  injected twice;
- re-ranking of `fact_store` search results with the embedding channel
  (`transform_tool_result`);
- post-session indexing of facts written since the last run (`on_session_end`).

Writes are never automatic. Conflict, duplicate and TTL actions remain explicit
review/apply operations on the store.

## Requirements

- Hermes Agent with the `holographic` memory provider enabled and an existing
  `~/.hermes/memory_store.db`.
- An OpenAI-compatible embeddings endpoint. The default is a local server at
  `http://127.0.0.1:1234/v1` (LM Studio, llama.cpp server, vLLM, Ollama's
  OpenAI-compatible API). Nothing is uploaded anywhere else.
- Python standard library only — no extra pip dependencies.

## Install

```bash
hermes plugins install MohamedElsiddig/hermes-plugin-factkit
```

That clones the repo into `~/.hermes/plugins/factkit`, records the Git source
and the manifest's `update_url`, and creates `config.yaml` from
`config.yaml.example` (never overwriting an existing one). Review the files
Hermes prints before confirming; pass `--no-enable` to install disabled, or
`--ref <40-char-sha>` to pin an exact commit.

## Enable

```bash
hermes plugins enable factkit
hermes plugins list | grep factkit        # status: enabled
```

Restart the gateway if it is already running so the hooks are registered.

## Configure

Edit `~/.hermes/plugins/factkit/config.yaml` (mode `600` is recommended because
it can hold an API key):

```yaml
embed_base_url: http://127.0.0.1:1234/v1
embed_api_key: ""
embed_model: text-embedding-nomic-embed-text-v1.5
prefetch: true
limit: 3
min_score: 0.55
```

| key | default | meaning |
|---|---|---|
| `embed_base_url` | `http://127.0.0.1:1234/v1` | OpenAI-compatible embeddings endpoint |
| `embed_api_key` | `""` | bearer token, if the endpoint requires one |
| `embed_model` | `text-embedding-nomic-embed-text-v1.5` | must match the model used to index the store |
| `prefetch` | `true` | inject the semantic block |
| `limit` | `3` | max facts injected per turn (clamped to 1–10) |
| `min_score` | `0.55` | minimum cosine similarity |
| `db` | `~/.hermes/memory_store.db` | store path |
| `merge_results` | `true` | fuse `fact_store` search results with embeddings |
| `semantic_weight` | `0.5` | embedding weight in the fused ranking |
| `embed_timeout_ms` | `1500` | per-turn budget; on timeout the turn proceeds without recall |
| `breaker_seconds` | `300` | cooldown after three consecutive endpoint failures |

Environment variables override the file: `FACTKIT_EMBED_BASE_URL`,
`FACTKIT_EMBED_API_KEY`, `FACTKIT_EMBED_MODEL`, `FACTKIT_DB`,
`FACTKIT_PREFETCH`, `FACTKIT_PREFETCH_LIMIT`, `FACTKIT_PREFETCH_MIN`,
`FACTKIT_MERGE`, `FACTKIT_INDEX`.

The hook fails soft: endpoint down, missing table, unreadable config or a bad
key returns no context instead of breaking the turn.

## Verify

```bash
hermes plugins list | grep factkit         # enabled
tail -f ~/.hermes/logs/gateway.log         # "factkit injected N fact(s)" at debug level
```

Embeddings must exist before recall returns anything. Facts written while the
plugin is enabled are indexed at session end automatically; to index the whole
existing store once, run an embedding pass with the factkit CLI (from the
[fact-memory-kit](https://github.com/MohamedElsiddig/fact-memory-kit) repo) or
let a few sessions populate new facts.

## Update

Manual update (the check is read-only; the update is explicit):

```bash
hermes plugins check-updates
hermes plugins update factkit --yes
```

`check-updates` reads this repo's `update.yaml` feed, compares its `version`
with the `version` in the installed `plugin.yaml`, and reports one of
`up to date`, `update available`, `unknown`, or `needs fixing`. `plugins update`
pulls the recorded Git source, runs the security scan, re-applies the manifest
validation and keeps your `config.yaml` and other untracked files.

### Auto-update (opt-in)

The manifest ships an `update_url`:

```
https://raw.githubusercontent.com/MohamedElsiddig/hermes-plugin-factkit/main/update.yaml
```

Hermes saves that URL at install time and refuses to follow a different one
that appears later unless you run `hermes plugins trust-update-url factkit`
after reviewing the change — that is the protection against a pulled commit
silently redirecting updates.

**Hermes never auto-updates this or any plugin unless you schedule it.** A
check costs one HTTPS request; applying an update rewrites files under
`~/.hermes/plugins/factkit`, so it is left to you.

Weekly check with cron (Monday 09:00), read-only:

```cron
0 9 * * 1 $HOME/.local/bin/hermes plugins check-updates >> $HOME/.hermes/logs/plugin-updates.log 2>&1
```

Weekly check-and-apply, if you want updates unattended:

```cron
0 9 * * 1 $HOME/.local/bin/hermes plugins update factkit --yes >> $HOME/.hermes/logs/plugin-updates.log 2>&1
```

Run `mkdir -p ~/.hermes/logs` first, and adjust the path if `command -v hermes`
points somewhere other than `~/.local/bin/hermes`. The equivalent Hermes cron
job is:

```bash
hermes cron create "0 9 * * 1" \
  "Run 'hermes plugins check-updates' and report whether factkit has an update; do not apply it." \
  --name factkit-update-check --deliver origin
```

### For maintainers

A release is two edits in one commit:

1. bump `version` in `plugin.yaml`;
2. bump `version` and `released` in `update.yaml`.

The feed's `artifacts.git` points at the repository, so the update path is a
plain fast-forward pull of `main`. Users on an older manifest see the new
version on their next check.

## Uninstall

```bash
hermes plugins disable factkit
hermes plugins remove factkit
```

`remove` deletes `~/.hermes/plugins/factkit`, including its `config.yaml`. The
memory database is untouched. To remove only the plugin's local edits first,
copy `config.yaml` somewhere else if you want to keep the settings.

## Files

```
plugin.yaml            manifest (name, version, hooks, repository, update_url)
__init__.py            plugin entry point: register() + the three hooks
factkit/holo.py        store access, prefetch, fusion, pending indexing
factkit/{config,embed,vectors,units,maintenance}.py
config.example.yaml    settings template (also shipped as config.yaml.example,
                       which Hermes copies to config.yaml on install)
update.yaml            auto-update feed read by `hermes plugins check-updates`
```

## Notes

- Injected facts are labelled `<semantic-memory>` and marked as data, never as
  instructions.
- Fact text stays in the local SQLite store; only the query text and fact text
  being embedded are sent to the configured embedding endpoint.
- No telemetry, no network calls other than your embedding endpoint and the
  update feed/`git pull` you ask for.
- The plugin never writes to the `facts` table from a hook; `on_session_end`
  only adds embeddings for facts that already exist.

## License

MIT — see [LICENSE](LICENSE).
