#!/usr/bin/env fish
# XLA_FLAGS auto-configuration for heterodyne (fish shell)
#
# This script configures XLA_FLAGS environment variable for optimal
# JAX CPU execution.
#
# Usage:
#   source xla_config.fish [mode]
#
# Modes:
#   auto  - Auto-detect based on CPU cores (default)
#   nlsq  - Single device for NLSQ fitting
#   cmc   - 4 devices for CMC sampling
#   cmc-hpc - 8 devices for HPC CMC
#   <N>   - Explicit number of devices

# Configuration file for persistent mode
set -g _HETERODYNE_XLA_MODE_FILE "$HOME/.heterodyne_xla_mode"

# Get CPU core count
function _heterodyne_get_cpu_count
    if command -v nproc > /dev/null 2>&1
        nproc
    else if command -v sysctl > /dev/null 2>&1
        sysctl -n hw.physicalcpu 2>/dev/null; or sysctl -n hw.ncpu 2>/dev/null; or echo 4
    else
        echo 4
    end
end

# Configure XLA_FLAGS
function _heterodyne_configure_xla
    set -l mode $argv[1]
    test -z "$mode"; and set mode "auto"

    set -l device_count

    switch $mode
        case nlsq
            set device_count 1
        case cmc
            set device_count 4
        case cmc-hpc
            set device_count 8
        case auto
            set -l cpu_count (_heterodyne_get_cpu_count)
            if test $cpu_count -lt 8
                set device_count $cpu_count
            else
                set device_count 8
            end
        case '*'
            # Check if numeric
            if string match -qr '^\d+$' $mode
                set device_count $mode
            else
                echo "Unknown XLA mode: $mode" >&2
                return 1
            end
    end

    # Don't override existing XLA_FLAGS if preserve flag is set
    if set -q HETERODYNE_PRESERVE_XLA_FLAGS; and set -q XLA_FLAGS
        return 0
    end

    # Build XLA_FLAGS
    set -l new_flag "--xla_force_host_platform_device_count=$device_count"

    if not set -q XLA_FLAGS
        set -gx XLA_FLAGS $new_flag
    else if not string match -q "*xla_force_host_platform_device_count*" $XLA_FLAGS
        set -gx XLA_FLAGS "$XLA_FLAGS $new_flag"
    end

    # Set JAX platform to CPU
    if not set -q JAX_PLATFORMS
        set -gx JAX_PLATFORMS cpu
    end
end

# Save mode to config file
function _heterodyne_save_xla_mode
    set -l mode $argv[1]
    echo $mode > $_HETERODYNE_XLA_MODE_FILE
end

# Load mode from config file
function _heterodyne_load_xla_mode
    if test -f $_HETERODYNE_XLA_MODE_FILE
        cat $_HETERODYNE_XLA_MODE_FILE
    else
        echo "auto"
    end
end

# Main entry point
function _heterodyne_xla_setup
    set -l mode

    if test (count $argv) -gt 0
        set mode $argv[1]
        _heterodyne_save_xla_mode $mode
    else
        set mode (_heterodyne_load_xla_mode)
    end

    _heterodyne_configure_xla $mode
end

# Run setup if argument provided
if test (count $argv) -gt 0
    _heterodyne_xla_setup $argv[1]
else
    _heterodyne_xla_setup
end
