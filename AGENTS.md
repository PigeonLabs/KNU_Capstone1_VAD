# Project working instructions

- Use only IPAD R01–R04 for the current approved experiments. Stage 1 is paper reproduction; stage 2 is DINOv2. Existing memory-removal runs are stage-1 ablations, not a new stage 3.
- Preserve raw data and all existing results. Record configuration, seed, environment/source hashes, commands, failures/restarts, training history, and aligned frame-level scores.
- Publication policy updated again 2026-09-25: the user authorizes automatic commits and pushes to main at https://github.com/PigeonLabs/KNU_Capstone1_VAD after each completed approved experiment (4-1, 4-2, 4-3, 5-1, 5-2, 5-3, 6-1, 6-2, 7-1, 7-2, 7-3). Report the result; no extra per-experiment approval is required. Keep Korean README evidence-based. Never force-push or overwrite unrelated remote work.
- Publish analysis artifacts and SHA256 inventories only. Do not upload dataset images, checkpoints, feature-cache binaries, local documents, credentials, or virtual environments. Large binaries remain local.
- Stop this project's experiment processes if disk free space reaches 10 GiB or below; report the pause. Never automatically resume a disk-space pause. Do not delete results to free space without user direction.
- Distinguish running, diagnostic, complete, and failed experiments. Do not claim exact paper equivalence: official source parameter count and underspecified details differ; R02 has excluded misaligned videos.
- The user approved stage 3 on 2026-09-25: normal-only phase diagnostics and fixed routing comparisons first, followed by evidence-based temporal diagnostics. Follow docs/stage3_protocol.md. Seed 1/2 repetitions follow diagnosis and configuration freeze, not silent test-set tuning.

- The user approved experiments 4-1, 4-2, and 4-3. Follow docs/stage4_protocol.md; preserve evaluation support and do not select settings using test AUROC.

- The user approved experiments 5-1, 5-2, 5-3 and confirmed current GPU / single stream 30 FPS. Follow docs/stage5_protocol.md; preserve disjoint normal calibration and causal inference.

- The user approved 6-1 Pareto efficiency and 6-2 normal-only calibration/generalization experiments. Follow docs/stage6_protocol.md, publish automatically on completion, and do not choose calibration using test labels.

- The user approved all recommended stage-7 experiments. Follow docs/stage7_protocol.md: diagnose first, preregister the evidence-based stage-7-2 choice, and validate full batch-one inference. Publish each completed stage automatically.

- The user approved 8-1/8-2/8-3 normal-only DINOv2 LoRA experiments. Follow docs/stage8_protocol.md, freeze settings before test evaluation, rebuild downstream models for every representation, and publish each completed stage automatically. No new full patch feature caches.
