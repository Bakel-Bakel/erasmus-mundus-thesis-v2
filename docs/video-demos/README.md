# Tiny video demos (Git exception folder)

Videos elsewhere in the repo are ignored by `.gitignore` (see root patterns like `*.mp4`, `*.mov`, …) so large binaries are not committed by accident.

This folder is **the only exception**: everything under `docs/video-demos/` is tracked by git.

**Only put very small files here** (for example under about 5 MB): short loops for the README, compressed web demos, etc. Prefer common formats (`.mp4`, `.webm`, `.mov`).

Do not use this folder for full-resolution survey footage; host large assets externally or keep them under `data/` (also gitignored).
