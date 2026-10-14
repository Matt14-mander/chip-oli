# OLI HU_D04 gripper assets

Copied from the local ADAPT OLI migration, retaining source files and licenses.
Official description revision: `80fbd2d650c43c38916979c2a6eab7a4c79e38de`.
Official deployment parameter revision: `bc76b0fe8399ecb84ce5492a6c385feb7d7a17a5`.

This is the URDF equivalent serial model, with eight gripper joints fixed at
zero and 31 body joints including the head. It does not reproduce the native
closed-chain ankle/waist motor-coordinate transmission. Contact is enabled
against terrain, with self collision disabled. Gains, armature and action
scales are recorded in `robot.json`; original assets and licenses are retained
in `source/` and `LICENSE`.
