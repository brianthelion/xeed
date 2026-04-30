`xeed` is an extremely compact project bootstrap — "seed" — that only depends on Python stdlib.

`xeed.d/` holds the configuration files that make `xeed` both compact and extensible.

# Quick start
```
wget -qO- https://raw.githubusercontent.com/brianthelion/xeed/main/xeed | python3
```

This bootstraps `xeed` and its minimal config into the current directory. From there:
```
xeed self/help   # list available commands
xeed self/pull   # pull the latest canonical xeed
```

# Self-management

`xeed` manages its own updates via the GitHub API — no git, no Docker required.

## xeed self/pull

Pulls the latest canonical `xeed` into the current project.

1. Compares local canonical files against the version at `XEED_ORIGIN_HASH`
2. If any local modifications are detected: exits with an error — run `xeed self/push` first
3. Otherwise: overwrites local `xeed` and all non-dunder `xeed.d/*.cfg` files with HEAD, and updates `XEED_ORIGIN_HASH` in the script

## xeed self/push

Pushes local modifications to the canonical xeed repo as a PR.

```
xeed self/push <branch> "<commit message>"
```

1. Detects which canonical files differ from `XEED_ORIGIN_HASH`
2. If nothing has changed: exits cleanly
3. Otherwise: requires both a branch name and commit message, then creates the branch (or pushes to it if it already exists), commits the changed files, and opens a PR

## xeed self/install

Installs an additional `xeed.d/` config from the canonical repo:

```
xeed self/install docker
```

## File conventions

**Dunder files** — any file in `xeed.d/` matching `__*.cfg` is project-local and never upstreamed by `self/push`. Use these for project-specific config.

The `xeed` script itself and all non-dunder `xeed.d/*.cfg` files are canonical and always upstreamed.

# Configuration

`xeed.d/*.cfg` files use standard `configparser` syntax with a modified placeholder format: `{section.option}` instead of `${section:option}`. Placeholders are resolved recursively, so a value that resolves to a string containing `{...}` will be re-resolved.

Multiline values use a `: ` prefix on each line:

```ini
[my.tool]
code:
    : print("hello")
    : print("world")
```
