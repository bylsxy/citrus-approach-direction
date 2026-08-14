# Citrus Approach-Direction Research Archive

Private working archive for the CEA citrus approach-direction paper and related Elfin experiments.

## Contents

- `data/elfin_citrus_data/`: complete CEA/Elfin data root, including sealed NBV batches, formal probe records, invalidated-run audit trails, calibration assets, models, paper results, and historical backups.
- `media/original_images/`: original photographs and screenshots used to build manuscript figures.
- `runtime/`: vision logs, freedrive trials, payload calibration runs, and current runtime parameter snapshots.
- `evidence/`: additional RGB-D and migration evidence.
- `notes/`: research decisions that existed only on this workstation.
- `operations/`: top-level Elfin launch, stop, recovery, and verification entry points.
- `environment/`: selected Noetic migration wrappers, manifests, handoff notes, and reproducibility scripts. Credentials and complete root filesystems are excluded.
- `manifests/SHA256SUMS.tsv`: path, byte size, and SHA-256 for every archived file.

## Related repositories

- Manuscript workspace: `https://github.com/bylsxy/paper-writing-workspace`
- Vision and Dashboard code: `https://github.com/bylsxy/elfin-citrus-harvest-vision`
- Panel and robot code: `https://github.com/bylsxy/elfin-ros-noetic-freedrive`

## Windows checkout

Use Git for Windows, then run:

```powershell
git config --global core.longpaths true
git clone https://github.com/bylsxy/citrus-approach-direction.git
```

The repository is private. Authenticate with the GitHub account that owns or has access to it. The archive contains immutable binary research data, so the first clone is large.

## Safety boundary

This repository contains research data and reproducibility material. It intentionally excludes SSH keys, API tokens, account databases, browser profiles, caches, third-party installations, and rebuildable Catkin build/devel directories.
