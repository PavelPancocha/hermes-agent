# BezKlikani Mem0 REST provider

Bundled fleet variant adapted from Mem0 vendor commit
`43849c6e9d4d773b83a8144b6373aedc67241a5d`, source path
[`integrations/hermes-plugin-mem0`](https://github.com/mem0ai/mem0/tree/43849c6e9d4d773b83a8144b6373aedc67241a5d/integrations/hermes-plugin-mem0).
The vendor LICENSE is preserved, including its upstream attribution.
This is an intentional fleet fork exception to Hermes' external-provider policy.

This variant supports self-hosted REST only, using Hermes' existing HTTPX dependency.
It installs no Mem0 SDK or other packages. Cloud and OSS backends, their setup
pickers/helpers, and the vendor compatibility-schema duplicates are omitted.

Local changes:

- Validate mode, HTTP(S) host, key and configured user/agent before client construction.
  Reject URL credentials, query and fragment. Preserve the configured `/api` prefix.
- Accept `selfhosted`, `self_hosted`, `self_hosted_http`, `rest`, `http`, and
  `platform` with a host. Normalize mode only for comparison; initialization does
  not migrate or rewrite existing `mem0.json` files or identities.
- Bind each provider instance to its owning profile home and scoped credentials.
  Configured literal `hermes-user` remains that exact principal. Search/list read
  across agents under that user; add retains the configured agent and channel.
  Update/delete require GET to return that exact user, including for legacy records.
- Restore `mem0_list`, bounded to the first **1,000 records**, with pages of 1–200
  (default 100). It is not a complete export: page 6 at size 200 is empty, and
  `count` describes the returned server window. Larger audits need server pagination.
- Retain vendor circuit breaking, validation, automatic turn extraction and its
  message truncation limit, profile-aware prefetch/sync threads, and bounded shutdown waits.
  Shutdown joins each background thread for up to five seconds and warns if it
  remains alive; unfinished sync may be lost when the backend closes.
  Four reviewed per-line code-health exceptions preserve the vendor catch-all error
  policies; Python type syntax is updated for the target lint rules.
- Retain the existing setup command, adding agent ID, `--api-url`, equals-form flags,
  strict provider flag validation (including unknown single-dash flags) and a dry
  run without writes or network probes. Provider help returns before setup starts.

Setup with synthetic placeholders (supply your actual scoped key privately):

```sh
hermes memory setup mem0 --mode self_hosted_http \
  --host https://memory.example/api --api-key SYNTHETIC_KEY \
  --user-id configured-user --agent-id configured-agent
```

Setup saves host and exact IDs in `mem0.json`, sets `mode=self_hosted_http` and
`rerank=false`, and saves a new key only in `.env`. Unknown JSON fields survive;
an unchanged legacy inline key can remain. A new `.env` key clears the old inline
field. Profile-scoped `MEM0_*` values remain defaults for nonempty file values.

Dry run validates and prints a sanitized summary. Normal setup writes `.env`, then
`mem0.json`, then activation in `config.yaml`; a failure propagates without newly
activating the provider. These writes are not one transaction: an earlier successful
write remains if a later one fails. A best-effort `/docs` probe proves reachability
only; it does not prove authentication or server readiness. Presence of this bundled
provider does not activate memory for a profile whose selection is disabled.
