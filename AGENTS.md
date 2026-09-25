# Project working instructions

- Use only IPAD R01–R04 for the current approved experiments. Stage 1 is paper reproduction; stage 2 is DINOv2. Existing memory-removal runs are stage-1 ablations, not a new stage 3.
- Preserve raw data and all existing results. Record configuration, seed, environment/source hashes, commands, failures/restarts, training history, and aligned frame-level scores.
- The user authorizes commits and pushes to `main` at `https://github.com/PigeonLabs/KNU_Capstone1_VAD` after each completed experiment stage. Keep the Korean README updated from evidence. Never force-push or overwrite unrelated remote work.
- Publish analysis artifacts and SHA256 inventories only. Do not upload dataset images, checkpoints, feature-cache binaries, local documents, credentials, or virtual environments. Large binaries remain local.
- Stop this project's experiment processes if disk free space reaches 10 GiB or below; report the pause. Never automatically resume a disk-space pause. Do not delete results to free space without user direction.
- Distinguish running, diagnostic, complete, and failed experiments. Do not claim exact paper equivalence: official source parameter count and underspecified details differ; R02 has excluded misaligned videos.
- The user approved stage 3 on 2026-09-25: normal-only phase diagnostics and fixed routing comparisons first, followed by evidence-based temporal diagnostics. Follow docs/stage3_protocol.md. Seed 1/2 repetitions follow diagnosis and configuration freeze, not silent test-set tuning.
