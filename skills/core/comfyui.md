# ComfyUI Skill (Layer 2)

This is the **OpenMontage-specific** guide to the ComfyUI generation provider
(`tools/graphics/comfyui_image.py`, `tools/video/comfyui_video.py`, shared
client in `tools/_comfyui/`). It explains when to pick ComfyUI over the
built-in CUDA local-gen tools (`wan_video`, `cogvideo_video`, etc.) or a paid
cloud provider, how to point it at a specific server, and what's actually
loaded on the reference deployment used during initial setup.

## What this is

A thin REST client against a running ComfyUI server (`tools/_comfyui/client.py`):
`POST /prompt` → poll `GET /history/{id}` → `GET /view`. No API key. Cost is
always `$0.0` (`estimate_cost` returns `0.0` on both tools) — this is a
self-hosted backend, never a billed one, and should never trip the budget
governance system's cost cap.

## When to pick ComfyUI vs the alternatives

| Situation | Prefer |
|---|---|
| GPU has ≥16GB VRAM available on the ComfyUI host | **ComfyUI** — bundled WAN 2.2 14B FP8 workflow is the highest local quality OpenMontage ships |
| GPU has 8GB VRAM and only the local host (no remote ComfyUI server) | **`wan_video` / `cogvideo_video`** in principle, but see the VRAM warning below — in practice these diffusers models frequently OOM on 8GB cards even though their declared `vram_mb` suggests they fit. Prefer routing to a ComfyUI server with more headroom when one is reachable. |
| Need a specific community workflow (ControlNet, LoRA stack, upscaling chain) | **ComfyUI** via `workflow_json`/`workflow_path` + `output_node` — both tools accept a fully custom API-format workflow |
| No ComfyUI server reachable anywhere on the network | Fall back to `fallback_tools` (`wan_video`, `hunyuan_video`, `ltx_video_local`, `kling_video` for video; `flux_image`, `local_diffusion`, `openai_image` for images) |

### Hard VRAM lesson (found during NUC13 setup, 2026-07)

The RTX 4060's 8188MiB nominal VRAM leaves only ~7.62GiB usable once the
driver/desktop reserve their share. Both `wan_video` (`wan2.1-1.3b`,
declared `vram_mb: 8000`) and `cogvideo_video` (`cogvideo-2b`, declared
`vram_mb: 6000`) OOM'd on this exact card during real smoke tests, even with
`enable_model_cpu_offload()` active (the tools' default) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set. The declared
`vram_mb` in `tools/video/_shared.py`'s `WAN_VARIANTS`/`COGVIDEO_VARIANTS` is
a **floor for weights residency**, not a promise the full generation loop
(text encoder + transformer + VAE decode + activation memory) fits. Treat an
8GB local card as "can attempt only the very lightest configs, expect OOM as
the common case" rather than as a working tier. This is exactly why the
ComfyUI-on-a-bigger-GPU path exists — don't burn time re-debugging diffusers
OOMs on an 8GB card before checking whether a remote ComfyUI server with more
headroom is reachable.

## Configuration

Set in `.env` (never hardcode in tool logic — both tools resolve this via
`os.environ.get("COMFYUI_SERVER_URL", "http://localhost:8188")` in
`ComfyUIClient.__init__`):

```
COMFYUI_SERVER_URL=http://192.168.1.88:8188
```

Note the env var is `COMFYUI_SERVER_URL`, **not** `COMFYUI_BASE_URL` — check
`tools/_comfyui/client.py` before trusting older notes that say otherwise.

`ComfyUIClient.is_available()` does a 5s-timeout `GET /system_stats` health
check. If the server is unreachable, `execute()` fails fast with
`unavailable_reason()` — a clear message telling the user to check
`COMFYUI_SERVER_URL` — rather than hanging the pipeline.

## Reference deployment (Ivan, CT 207 — verified 2026-07-15)

- `http://192.168.1.88:8188`, AMD Radeon RX 7900 XT, ~20GB VRAM (~18.4GB free
  at idle), ROCm, gfx1100 (RDNA3). No auth, LAN-only.
- Runs as a Docker container named `comfyui` inside the LXC. If unreachable,
  it may be crash-looping — check `docker logs comfyui` for
  `RuntimeError: No HIP GPUs are available`, which indicates a host-level
  GPU/driver problem, not anything wrong with the OpenMontage side. SSH
  access and container map: see the `ssh-ivan` skill.
- **Loaded checkpoints (image, SD-family):** `juggernautXL_v9.safetensors`,
  `realisticVisionV60B1_v60B1VAE.safetensors`, `v1-5-pruned-emaonly.safetensors`.
  These work today via a **custom** `workflow_json` (see below) — the
  bundled `flux2-txt2img.json` workflow does NOT match these checkpoints
  (FLUX uses `UNETLoader`/`DualCLIPLoader`, not `CheckpointLoaderSimple`).
- **Video models: none installed.** `models/diffusion_models`, `models/unet`,
  `models/text_encoders`, `models/vae` are all empty (just ComfyUI's default
  `put_..._here` placeholder files). The bundled `wan22-t2v-4step.json` /
  `wan22-i2v-4step.json` workflows reference WAN 2.2 14B FP8 weights that
  are not present, so `comfyui_video` will return `success=False` with
  `data.missing_models` (see `_REQUIRED_MODELS_T2V`/`_REQUIRED_MODELS_I2V`
  in `tools/video/comfyui_video.py`) until someone downloads them onto CT 207
  (`tools/_comfyui/metadata.py` → `BUNDLED_MODEL_STACKS` has the exact
  filenames and HuggingFace URLs). This is a real, sizable download
  (WAN 2.2 14B FP8 weights run tens of GB) — confirm with the user and their
  available disk/bandwidth on CT 207 before triggering it, it is not
  something to do silently.
- Node inventory confirms this instance has the full WAN family of custom
  nodes installed (`WanImageToVideo`, `WanVaceToVideo`, etc. — 30+ WAN node
  types in `/object_info`) plus Kling/ByteDance/Grok API wrapper nodes (those
  need their own API keys — irrelevant to the self-hosted use case here).

## Building a custom workflow_json

Both tools accept `workflow_json` (inline JSON string) or `workflow_path`
(file path) plus a required `output_node` (the node ID whose output image/
video should be downloaded). Use the standard ComfyUI **API format**
(`{"<node_id>": {"class_type": ..., "inputs": {...}}}`), not the UI/graph
export format. Minimal SD1.5 txt2img example, validated against CT 207:

```python
workflow = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
        "scheduler": "normal", "denoise": 1.0,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0],
    }},
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
    "5": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "<prompt>", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "<negative>", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode",
          "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage",
          "inputs": {"filename_prefix": "om", "images": ["8", 0]}},
}
# output_node = "9"
```

Node input arrays like `["4", 1]` mean "output slot 1 of node 4" — this is
how ComfyUI wires the graph in API format. Use `ComfyUIClient.list_models()`
or `GET /object_info/<NodeClass>` to discover exactly which checkpoint/vae/
clip filenames a given server actually has before hardcoding one into a
workflow — don't assume the bundled workflow's model names apply to a
different ComfyUI instance.

## Provenance

Every successful result includes `data.workflow_provenance` — `source`
(`bundled` vs `user_supplied`), `workflow_hash_sha256`, and `model_stack`.
For custom workflows, pass `workflow_name` / `workflow_model` /
`workflow_model_stack` in the inputs so provenance isn't `unknown_custom_workflow`.

## Registry / discovery

Both tools are picked up automatically by `tools/tool_registry.py`'s
directory-scan discovery — no manual registration needed. Confirm via:

```bash
.venv/bin/python -c "from tools.tool_registry import registry; import json; registry.discover(); print(json.dumps(registry.provider_menu(), indent=2))"
```

`comfyui_image` and `comfyui_video` should appear alongside `flux_image`,
`wan_video`, etc.
