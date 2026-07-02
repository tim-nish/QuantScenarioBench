#!/usr/bin/env bash
set -euo pipefail

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Link shared Claude Code commands (bind-mounted from host, read-only)
mkdir -p "$HOME/.claude"
ln -sfn /opt/claude-commands "$HOME/.claude/commands"

# Install project (including dev/test dependencies) in editable mode
if [ -f pyproject.toml ]; then
  pip install --user -e ".[dev]"
fi
