#!/bin/bash
# XLA_FLAGS auto-configuration for heterodyne (bash/zsh compatible)
#
# This script configures XLA_FLAGS environment variable for optimal
# JAX CPU execution. Source this in your shell or venv activate script.
#
# Usage:
#   source xla_config.bash [mode]
#
# Modes:
#   auto  - Auto-detect based on CPU cores (default)
#   nlsq  - Single device for NLSQ fitting
#   cmc   - 4 devices for CMC sampling
#   cmc-hpc - 8 devices for HPC CMC
#   <N>   - Explicit number of devices

# Configuration file for persistent mode
_HETERODYNE_XLA_MODE_FILE="${HOME}/.heterodyne_xla_mode"

# Get CPU core count (cross-platform)
_heterodyne_get_cpu_count() {
    if command -v nproc &>/dev/null; then
        nproc
    elif command -v sysctl &>/dev/null; then
        sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
    else
        echo 4  # Fallback
    fi
}

# Configure XLA_FLAGS
_heterodyne_configure_xla() {
    local mode="${1:-auto}"
    local device_count

    case "$mode" in
        nlsq)
            device_count=1
            ;;
        cmc)
            device_count=4
            ;;
        cmc-hpc)
            device_count=8
            ;;
        auto)
            # Auto-detect: use min(physical_cores, 8)
            local cpu_count
            cpu_count=$(_heterodyne_get_cpu_count)
            device_count=$((cpu_count < 8 ? cpu_count : 8))
            ;;
        [0-9]*)
            # Explicit number
            device_count="$mode"
            ;;
        *)
            echo "Unknown XLA mode: $mode" >&2
            return 1
            ;;
    esac

    # Don't override existing XLA_FLAGS if set by user
    if [[ -n "${HETERODYNE_PRESERVE_XLA_FLAGS:-}" && -n "${XLA_FLAGS:-}" ]]; then
        return 0
    fi

    # Build XLA_FLAGS
    local new_flag="--xla_force_host_platform_device_count=${device_count}"

    if [[ -z "${XLA_FLAGS:-}" ]]; then
        export XLA_FLAGS="$new_flag"
    elif [[ "$XLA_FLAGS" != *"xla_force_host_platform_device_count"* ]]; then
        export XLA_FLAGS="${XLA_FLAGS} ${new_flag}"
    fi

    # Also set JAX platform to CPU
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
}

# Save mode to config file
_heterodyne_save_xla_mode() {
    local mode="$1"
    local tmp_file
    tmp_file=$(mktemp "${_HETERODYNE_XLA_MODE_FILE}.XXXXXX")
    echo "$mode" > "$tmp_file"
    mv "$tmp_file" "$_HETERODYNE_XLA_MODE_FILE"
}

# Load mode from config file
_heterodyne_load_xla_mode() {
    if [[ -f "$_HETERODYNE_XLA_MODE_FILE" ]]; then
        cat "$_HETERODYNE_XLA_MODE_FILE"
    else
        echo "auto"
    fi
}

# Main entry point
_heterodyne_xla_setup() {
    local mode

    # Check for argument or load saved mode
    if [[ -n "${1:-}" ]]; then
        mode="$1"
        # Save if explicitly provided
        _heterodyne_save_xla_mode "$mode"
    else
        mode=$(_heterodyne_load_xla_mode)
    fi

    _heterodyne_configure_xla "$mode"
}

# Run if sourced with argument, or auto-configure
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Being sourced
    _heterodyne_xla_setup "${1:-}"
fi
