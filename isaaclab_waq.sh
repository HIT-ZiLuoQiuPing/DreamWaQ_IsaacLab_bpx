#!/usr/bin/env bash

export ISAACLAB_WAQ_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="${ISAACLAB_WAQ_PATH}/source/isaaclab_waq:${PYTHONPATH}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${ISAACLAB_WAQ_PATH}/.cache/warp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ISAACLAB_WAQ_PATH}/.cache/matplotlib}"

if ! [[ -z "${CONDA_PREFIX}" ]]; then
    python_exe=${CONDA_PREFIX}/bin/python
else
    echo "[Error] No conda environment activated. Please run: conda activate isaaclab_bpx"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--install)
            pip install -e "${ISAACLAB_WAQ_PATH}/source/isaaclab_waq"
            break
            ;;
        -wt|--waq-train)
            shift
            "${python_exe}" "${ISAACLAB_WAQ_PATH}/scripts/waq/train.py" "$@"
            break
            ;;
        -wp|--waq-play)
            shift
            "${python_exe}" "${ISAACLAB_WAQ_PATH}/scripts/waq/play.py" "$@"
            break
            ;;
        -we|--waq-export)
            shift
            "${python_exe}" "${ISAACLAB_WAQ_PATH}/scripts/waq/export_policy.py" "$@"
            break
            ;;
        -mp|--mujoco-play)
            shift
            "${python_exe}" "${ISAACLAB_WAQ_PATH}/scripts/sim2sim/mujoco_play.py" "$@"
            break
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            echo "Usage: ./isaaclab_waq.sh --install | --waq-train [args] | --waq-play [args] | --waq-export [args] | --mujoco-play [args]"
            exit 1
            ;;
    esac
done
