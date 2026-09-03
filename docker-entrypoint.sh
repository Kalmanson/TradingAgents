#!/bin/sh
set -e

# Ensure the .tradingagents directory and subdirectories are writable by appuser.
# Docker named volumes are created with root:root ownership by default;
# when mounted over /home/appuser/.tradingagents the container user cannot
# write logs/reports/cache/memory.  Fix ownership at startup so every
# container start works regardless of prior volume state.
if [ "$(id -u)" = "0" ]; then
    # Running as root (e.g. docker exec --user root) — fix and drop privileges
    chown -R appuser:appuser /home/appuser/.tradingagents 2>/dev/null || true
    exec gosu appuser "$@"
else
    # Running as appuser (normal case) — try to create dirs; ignore failures
    # (host bind-mounts may already have correct ownership).
    mkdir -p /home/appuser/.tradingagents/logs \
             /home/appuser/.tradingagents/cache \
             /home/appuser/.tradingagents/memory 2>/dev/null || true
    exec "$@"
fi
