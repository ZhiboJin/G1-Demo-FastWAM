# Dataset selection for the staged training plan

| Priority | Source | Intended use | Checks before mixing |
|---|---|---|---|
| 1 | [OpenHLM/OpenHLM-data](https://huggingface.co/datasets/OpenHLM/OpenHLM-data), especially `g1_HLM-12_full_teleop` | Native G1 full-body action reference and initial posttraining set | Confirm its 34-value ordering, hand convention, image streams, frame rate, and task split from the actual subset metadata. |
| 2 | [nvidia/g1_locomanip_dataset](https://huggingface.co/datasets/nvidia/g1_locomanip_dataset) | Synthetic G1 pick, navigate, place task variety | Convert its policy/control representation to the chosen 34-D contract; tag as synthetic, and hold out scenes/objects. |
| 3 | [Unitree UnifoLM WBT collection](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) | Broader real G1 tasks after schema review | Collection has many separate datasets; choose subsets with compatible body/hand hardware, synchronized images, and reusable action labels. Check each license. |
| Optional | [OpenHLM HuMI and stationary subsets](https://github.com/OpenHLM-project/OpenHLM) | Heterogeneous embodiment augmentation | Keep embodiment masks and source labels. Do not treat stationary-arm data as full-body supervision. |

The existing FastWAM [LIBERO/RoboTwin data and checkpoints](https://github.com/yuantianyuan01/FastWAM) are useful for pipeline validation and weight initialization experiments, but their seven-dimensional action statistics cannot be reused as G1 normalization.

## Common episode format to implement next

Each timestep: `timestamp`, `instruction`, `head_rgb`, optional `left_wrist_rgb` and `right_wrist_rgb`, `joint_position[29]`, `joint_velocity[29]`, `base_orientation`, `action[contract_dim]`, `action_valid_mask[contract_dim]`, `embodiment_id`, `task_id`, `episode_id`, `source`. Use the contract order for all actions. Preserve the source action and calibration metadata in sidecar files. Resample only after checking whether commands are positions, deltas, velocities, or torques. Split by task/scene and episode before computing per-embodiment statistics.
