# Inspire camera0 frame 676 - self-contained URDF visualization

This directory is path-independent. It contains the final GLB, transparent PNG, sanitized frame data, visual-only URDF, every referenced visual GLB, and scripts.

## Files

- `model/inspire_hand_object.glb`: final self-contained GLB (no external URI).
- `preview/inspire_hand_object_transparent.png`: transparent camera0 render.
- `metadata/result.json`: relative-path generation and validation record.
- `source/`: bundled frame JSON, RGB, and depth.
- `assets/inspire_hand/`: visual-only URDF and original GLB/PBR meshes.
- `scripts/`: regeneration and independence checks.

## Verify

```bash
python scripts/verify_bundle.py
```

## Regenerate (GPU/EGL for the transparent preview)

```bash
python scripts/regenerate.py
```

Both scripts locate this directory from their own file path, so the bundle can be copied or renamed without editing paths.
