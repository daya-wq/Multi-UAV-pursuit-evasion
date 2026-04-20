# Project Log: Multi-UAV-pursuit-evasion

This document tracks all operations, environment setups, and modifications performed by the assistant.

## Project Structure Documentation (Latest)
- **Action**: Performed deep analysis of the entire codebase (~108 files), reading all key source files including `isaac_env.py`, `hideandseek.py` (1284 lines), `hideandseek_envgen.py` (1612 lines), `hideandseek_deploy.py` (1272 lines), `mappo.py` (662 lines), `multirotor.py` (759 lines), `robot.py`, `rotor_group.py`, all YAML configs, training scripts, and utility modules.
- **Action**: Created comprehensive project structure documentation at `docs/project_structure.md` (~500 lines), covering 9 major sections: project overview, directory tree, all `omni_drones/` subpackages, configuration system, training scripts, dependencies, environment setup, data flow diagram (Mermaid), and key technical points.
- **Status**: Documentation complete. All future docs will be placed under `docs/`.

## Initialization
- **Action**: Cloned clean repository from `https://github.com/thu-uav/Multi-UAV-pursuit-evasion`.
- **Action**: Fixed `.gitmodules` from SSH to HTTPS to allow cloning of submodules without SSH keys, and successfully initialized `tensordict` and `torchrl` submodules.
- **Action**: Configured global Git user credentials (`daya-wq`, `2625512566@qq.com`). Generated a new ED25519 SSH key and registered it with GitHub.
- **Action**: Tested SSH connection to GitHub successfully. Changed `origin` to the user's fork (`git@github.com:daya-wq/Multi-UAV-pursuit-evasion.git`) and retained official repository as `upstream`.
- **Status**: Environment and Version Control fully set up.
- **Next Steps**: Ready for any specific code modifications or test scripts as requested by the user.
