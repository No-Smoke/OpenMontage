# OpenMontage — NUC13 Operator Manual

Installed 2026-07-15. Two-page reference for running and maintaining this
instance. For the full framework docs see `README.md` and
`docs/ARCHITECTURE.md`.

## 1. What's installed

| Component | Location |
|---|---|
| Repo (upstream, read-only) | `/home/vanya/Projects/OpenMontage`, remote `origin` = `calesthio/OpenMontage` |
| Your fork (pushable) | remote `fork` = `github.com/No-Smoke/OpenMontage` |
| Python env | `.venv/` — uv-managed, Python 3.12.13 |
| Node | nvm-managed v22.14.0 (`~/.nvm/versions/node/v22.14.0/`) |
| Config | `.env` (gitignored — never committed) |

**Always activate the venv first:** `source .venv/bin/activate`, or prefix
commands with `.venv/bin/python`. This venv has **no `pip` binary** — use
`uv pip install <pkg> --python .venv/bin/python`, not plain `pip`.

## 2. Two generation backends

OpenMontage can generate video/images two ways. Both are configured in `.env`
and require no manual tool registration (auto-discovered by
`tools/tool_registry.py`).

### A. Local GPU (RTX 4060, 8GB VRAM) — `wan_video`, `cogvideo_video`, etc.

```
VIDEO_GEN_LOCAL_ENABLED=true
VIDEO_GEN_LOCAL_MODEL=cogvideo-2b
```

**Known limitation:** every diffusers-based local video model tested
(wan2.1-1.3b, cogvideo-2b) reliably **OOMs** on this 8GB card during actual
generation, despite documented 6-8GB minimums. The real usable VRAM is
~7.62GB (not the full 8GB), and peak generation memory exceeds that. Treat
local video gen as non-functional in practice; it's wired up but don't rely
on it. Local **image** generation (much lighter) is unaffected by this.

### B. ComfyUI on CT207 (Ivan, RX 7900 XT, 20GB VRAM) — PRIMARY video backend

```
COMFYUI_SERVER_URL=http://192.168.1.88:8188
```

This is a Docker container (`comfyui`) inside Proxmox LXC CT207 at
`192.168.1.79`. No auth, LAN-only. Confirmed working for both image and
video generation as of 2026-07-15 (WAN 2.2 T2V + I2V model weights
installed, ~190GB disk quota, 40GB RAM quota).

**Settled default model:** `wan2.1_t2v_1.3B_fp16.safetensors` via the custom
workflow `tools/_comfyui/workflows/wan21-1.3b-t2v.json` — fast (well under a
minute per clip), reliable, no VRAM/RAM pressure. The bundled WAN 2.2 14B
dual-expert workflow (`wan22-t2v-4step.json`) works but is NOT the default
recommendation: it loads ~28GB of combined weights against CT207's 20GB
VRAM, forcing heavy CPU-offload swapping that both slows generation to
30+ minutes per clip and previously OOM-killed the container (fixed by the
RAM bump below, but still slow). Use the 14B workflow only when quality
matters more than turnaround time.

See `skills/core/comfyui.md` for the full write-up: custom workflow_json
construction, model inventory, env var details, VRAM/RAM findings.

## 3. Common commands

```bash
source .venv/bin/activate

make test-contracts        # run the test suite (no API keys needed)
make preflight              # show every tool + its status (provider menu)
make demo                   # zero-key Remotion demo render
npx --yes hyperframes doctor   # check HyperFrames runtime health
```

## 4. If ComfyUI (CT207) is unreachable

```bash
curl http://192.168.1.88:8188/system_stats     # should return JSON
```

If it fails, the container may be crash-looping. Check via SSH (see
`ssh-ivan` skill for credentials):

```bash
ssh -i ~/.ssh/id_ed25519 root@192.168.1.79 \
  'pct exec 207 -- docker ps -a --filter name=comfyui'
ssh -i ~/.ssh/id_ed25519 root@192.168.1.79 \
  'pct exec 207 -- docker logs --tail 30 comfyui'
```

`RuntimeError: No HIP GPUs are available` in the logs means CT207's own
ROCm/GPU passthrough is broken — a host-level issue, not an OpenMontage
problem. Restart the container or escalate to Ivan's infra maintenance.

## 5. Disk / RAM capacity notes

CT207's LXC disk quota was resized **100GB → 190GB** on 2026-07-15 to fit
the WAN 2.2 model stack (~65.6GB: text encoder, T2V + I2V diffusion models,
VAEs, LoRAs). Its RAM quota was separately resized **24GB → 40GB** after the
bundled 14B dual-expert workflow's CPU-offload swapping OOM-killed the
ComfyUI process (kernel `dmesg` showed a memcg OOM kill at ~24.8GB resident
— Docker's `unless-stopped` policy silently restarted the container after,
which looks like an unrelated crash unless you check `dmesg`). Both resizes
apply live via `pct` with no container restart needed. Check headroom before
adding more models or trying heavier workflows:

```bash
ssh -i ~/.ssh/id_ed25519 root@192.168.1.79 'pct exec 207 -- df -h /; free -h'
ssh -i ~/.ssh/id_ed25519 root@192.168.1.79 'dmesg | grep -i "oom\|killed process" | tail -5'
```

## 6. Adding API keys

Edit `.env` directly — every provider (fal.ai, ElevenLabs, HeyGen, etc.) is
optional and the pipeline degrades gracefully without them, routing to
free/local providers where possible. No keys are required for the
ComfyUI or local-image paths.

## 7. Gotchas

- `ffmpeg` was missing at first install; confirm it's still on PATH:
  `ffmpeg -version`.
- `.env` is gitignored — it will never accidentally get pushed/forked.
- `origin` remote is the public upstream repo — do not push there. Use
  `fork` for any commits you want to keep (e.g. `git push fork main`).
- The repo's own `.python-version` says 3.10; this install uses 3.12
  (README says "3.10+", so this is within spec, just newer).
