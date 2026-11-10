# Lecture 1 helpers

| Module | Used by | Responsibility |
|---|---|---|
| `dh_demo.py` | `dh_forward_kinematics.py` | KUKA setup, DH frames, traces and pose comparison |
| `jacobian_geometry.py` | `jacobian_demo.py` | Reference circle from the joint axis and end-effector position |
| `jacobian_ik_ui.py` | `jacobian_ik_step.py` | Manual IK panel, target editing and diagnostics |

Run the classroom entries in `lecture1/`; these modules contain their reusable setup/display code. Cross-lecture utilities live in `scripts/shared/`.
