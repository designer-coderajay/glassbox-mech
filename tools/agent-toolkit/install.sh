#!/usr/bin/env bash
# Install the third-party agent toolkit used while building Glassbox.
#
# Clones every repo in repos.lock at its pinned commit into .agent-tools/
# (git-ignored), then wires the Claude Code pieces into the user-level
# config (~/.claude or $CLAUDE_CONFIG_DIR). Nothing is copied into the
# Glassbox source tree. Safe to re-run.
#
# Usage:
#   bash tools/agent-toolkit/install.sh [options]
#
# Options:
#   --only a,b              Only these tools (names from repos.lock)
#   --agency-divisions a,b  Only these agency-agents divisions (default: all)
#   --setup-mirofish        Also install MiroFish backend + frontend deps (uv, npm)
#   --reach-channels a,b    Also run `agent-reach install --channels a,b`
#                           (installs extra global CLIs via pipx/npm)
#   --clone-only            Clone/check out pinned commits, wire nothing
#   -h, --help              Show this help
#
# Env:
#   AGENT_TOOLKIT_HOME   Clone directory (default: <repo>/.agent-tools)
#   CLAUDE_CONFIG_DIR    Claude Code config root (default: ~/.claude)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK="$SCRIPT_DIR/repos.lock"
TOOLS_HOME="${AGENT_TOOLKIT_HOME:-$REPO_ROOT/.agent-tools}"
CLAUDE_HOME="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

ONLY=""
AGENCY_DIVISIONS=""
SETUP_MIROFISH=0
REACH_CHANNELS=""
CLONE_ONLY=0

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --agency-divisions) AGENCY_DIVISIONS="$2"; shift 2 ;;
    --setup-mirofish) SETUP_MIROFISH=1; shift ;;
    --reach-channels) REACH_CHANNELS="$2"; shift 2 ;;
    --clone-only) CLONE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*"; }

FAILED=()

selected() {
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]]
}

# Clone or move an existing clone to the pinned commit.
checkout_pinned() {
  local name="$1" url="$2" sha="$3" dir="$TOOLS_HOME/$1"
  if [[ ! -d "$dir/.git" ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]]; then
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
  ok "$name @ ${sha:0:12}"
}

# Symlink src into dest_dir/<basename> unless something else already lives there.
link_into() {
  local src="$1" dest_dir="$2" name; name="$(basename "$src")"
  mkdir -p "$dest_dir"
  local dest="$dest_dir/$name"
  if [[ -L "$dest" ]]; then
    ln -sfn "$src" "$dest"
  elif [[ -e "$dest" ]]; then
    warn "skip $dest (already exists, not ours)"
    return 1
  else
    ln -s "$src" "$dest"
  fi
}

install_agency_agents() {
  local args=(--tool claude-code --no-interactive --path "$CLAUDE_HOME/agents")
  [[ -n "$AGENCY_DIVISIONS" ]] && args+=(--division "$AGENCY_DIVISIONS")
  (cd "$TOOLS_HOME/agency-agents" && ./scripts/install.sh "${args[@]}")
}

# codex-first reroutes all implementation work to an external Codex CLI; it
# would hijack Claude Code sessions here, so it is never linked.
AGENT_SCRIPTS_SKIP=",codex-first,"

install_agent_scripts() {
  local n=0 d
  for d in "$TOOLS_HOME/agent-scripts/skills"/*/; do
    [[ -f "$d/SKILL.md" ]] || continue  # dangling links to steipete's other repos
    [[ "$AGENT_SCRIPTS_SKIP" == *",$(basename "$d"),"* ]] && continue
    link_into "${d%/}" "$CLAUDE_HOME/skills" && n=$((n + 1))
  done
  ok "agent-scripts: $n skills linked into $CLAUDE_HOME/skills"
}

install_cli_anything() {
  local root="$TOOLS_HOME/CLI-Anything"
  if command -v claude >/dev/null 2>&1; then
    claude plugin marketplace add "$root" >/dev/null 2>&1 || claude plugin marketplace update cli-anything >/dev/null
    claude plugin install cli-anything@cli-anything
  else
    warn "claude CLI not found; skipped the cli-anything plugin"
  fi
  uv tool install --force "$root/cli-hub" >/dev/null
  ok "cli-hub installed ($(command -v cli-hub || echo "$HOME/.local/bin/cli-hub"))"
}

install_agent_reach() {
  uv tool install --force --with-executables-from yt-dlp "$TOOLS_HOME/Agent-Reach" >/dev/null
  local bin; bin="$(command -v agent-reach || echo "$HOME/.local/bin/agent-reach")"
  # The packaged skill installer writes to ~/.claude/skills (and other agents' dirs).
  AGENT_REACH_LANG=en "$bin" skill --install
  if [[ -n "$REACH_CHANNELS" ]]; then
    "$bin" install --channels "$REACH_CHANNELS"
  fi
  ok "agent-reach installed ($bin); run 'agent-reach doctor' to see which channels work"
}

install_gmaps_kit() {
  local root="$TOOLS_HOME/google-maps-scraper-kit"
  link_into "$root/.claude/skills/google-maps-scraper" "$CLAUDE_HOME/skills" || true
  local c
  for c in "$root/.claude/commands"/*.md; do
    link_into "$c" "$CLAUDE_HOME/commands" || true
  done
  # The kit's own .claude/settings.json pre-approves docker/curl commands; it is
  # deliberately not merged into any settings here.
  ok "google-maps-scraper-kit: skill + /scrape commands linked (engine: docker compose up -d in $root)"
}

install_mirofish() {
  local root="$TOOLS_HOME/MiroFish"
  if [[ "$SETUP_MIROFISH" == 1 ]]; then
    (cd "$root" && npm run setup && npm run setup:backend)
    [[ -f "$root/.env" ]] || cp "$root/.env.example" "$root/.env"
    ok "MiroFish deps installed; fill in $root/.env then 'npm run dev' there"
  else
    ok "MiroFish cloned (run with --setup-mirofish to install its deps)"
  fi
}

mkdir -p "$TOOLS_HOME"
log "Cloning pinned repos into $TOOLS_HOME"
while read -r name url sha _license; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  selected "$name" || continue
  checkout_pinned "$name" "$url" "$sha" || FAILED+=("$name (clone)")
done < "$LOCK"

if [[ "$CLONE_ONLY" == 0 ]]; then
  log "Wiring into $CLAUDE_HOME"
  for name in agency-agents agent-scripts CLI-Anything Agent-Reach google-maps-scraper-kit MiroFish; do
    selected "$name" || continue
    [[ -d "$TOOLS_HOME/$name" ]] || continue
    case "$name" in
      agency-agents)           fn=install_agency_agents ;;
      agent-scripts)           fn=install_agent_scripts ;;
      CLI-Anything)            fn=install_cli_anything ;;
      Agent-Reach)             fn=install_agent_reach ;;
      google-maps-scraper-kit) fn=install_gmaps_kit ;;
      MiroFish)                fn=install_mirofish ;;
    esac
    log "$name"
    "$fn" || FAILED+=("$name")
  done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  warn "failed: ${FAILED[*]}"
  exit 1
fi
log "Done. Start a new Claude Code session to pick up new agents, skills and plugins."
