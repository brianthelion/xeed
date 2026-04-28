# xeed self-management design notes

## Problem

`xeed` and `xeed.d/` are embedded inside project repos (e.g. gnokin). They need to be independently revision-controlled against a canonical xeed repo, with the ability to push local xeed changes upstream and pull new xeed versions into projects.

## File conventions

**Dunder files** — any file in `xeed.d/` matching `__*__.cfg` is **not upstreamed**. It may be committed to the project repo normally. Everything else in `xeed.d/` is canonical and gets upstreamed by `self/commit`.

Key dunder files:
- `__xeed__.cfg` — install metadata; written on first run; contains `XEED_ORIGIN_HASH`
- Any project-specific config (e.g. `__project__.cfg`) — free to create, never upstreamed

The `xeed` script itself is always canonical and always upstreamed.

## Origin hash

The canonical repo URL and the commit hash of the installed version are the two anchors for all self-management operations.

**How the hash gets into a fresh install:**

`xeed` is distributed via `wget` pointing at a `git archive`-generated artifact (e.g. a GitHub release asset). The script contains:

```python
XEED_ORIGIN_HASH = "$Format:%H$"
```

With `.gitattributes`:

```
xeed export-subst
```

`git archive` substitutes the actual commit hash at export time. The downloaded file arrives with the hash baked in.

**On first run** (or as part of `xeed self/install`), xeed reads its own `XEED_ORIGIN_HASH` constant and writes it to `xeed.d/__xeed__.cfg`. All subsequent self-management operations read from there.

The canonical repo URL lives in `xeed.cfg` (canonical, same everywhere).

## `xeed self/commit`

Pushes local modifications to canonical xeed files upstream.

1. Read `XEED_ORIGIN_HASH` from `__xeed__.cfg`
2. In a Docker container with git:
   - Clone canonical repo
   - Checkout at `XEED_ORIGIN_HASH`
   - Apply diff between canonical@`XEED_ORIGIN_HASH` and current local canonical files as a new commit
   - Push to a branch
3. Dunder files are excluded by the `__*__.cfg` pattern — never upstreamed

## `xeed self/pull`

Pulls the latest canonical xeed into the project.

1. Diff local canonical files against canonical@`XEED_ORIGIN_HASH`
2. If any diff exists: **exit with error** — "local modifications detected, run `xeed self/commit` first"
3. Otherwise, in a Docker container with git:
   - Clone canonical HEAD
   - Overwrite local `xeed` script and all non-dunder `xeed.d/` files
   - Update `XEED_ORIGIN_HASH` in `__xeed__.cfg` to new HEAD hash

No merge logic in `self/pull`. Complexity lives entirely in `self/commit`.

## Intended workflow

```
# First time in a project
wget <canonical-url>/xeed
./xeed self/install     # writes __xeed__.cfg

# Pulling updates
xeed self/pull          # refuses if local canonical changes exist

# Contributing changes back
xeed self/commit        # branches from XEED_ORIGIN_HASH, pushes
xeed self/pull          # once changes are merged upstream, pull them back
```

## Bootstrap / chicken-and-egg notes

- Hash is embedded at **publish time** via `git archive` + `export-subst`, not at install time
- `wget` URL must point to a `git archive` artifact, not the raw repo file (raw files don't get `export-subst` substitution)
- `self/pull` uses Docker + git directly — no `wget` needed since xeed is already running
- The first real exercise of `self/pull` after building this tooling will be pulling it into embedded projects (self-validating)
