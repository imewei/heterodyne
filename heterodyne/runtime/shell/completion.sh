#!/bin/bash
# Bash completion for heterodyne CLI
#
# Installation:
#   Source this file in your .bashrc or copy to /etc/bash_completion.d/
#
# Features:
#   - Context-aware completions for options
#   - Config file caching (5-minute TTL)
#   - Method suggestions based on workflow

# Cache directory for completions
_HETERODYNE_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/heterodyne"
_HETERODYNE_CACHE_TTL=300  # 5 minutes

# Ensure cache directory exists
_heterodyne_ensure_cache() {
    [[ -d "$_HETERODYNE_CACHE_DIR" ]] || mkdir -p "$_HETERODYNE_CACHE_DIR"
}

# Fallback for _init_completion when bash-completion is not loaded
# (common in conda/mamba environments)
if ! type _init_completion &>/dev/null; then
    _init_completion() {
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    }

    # Fallback for _filedir (directory completion)
    if ! type _filedir &>/dev/null; then
        _filedir() {
            if [[ "$1" == "-d" ]]; then
                mapfile -t COMPREPLY < <(compgen -d -- "${cur}")
            else
                mapfile -t COMPREPLY < <(compgen -f -- "${cur}")
            fi
        }
    fi
fi

# Get cached config files or update cache
_heterodyne_get_config_files() {
    _heterodyne_ensure_cache
    local cache_file="$_HETERODYNE_CACHE_DIR/config_files"
    local now
    now=$(date +%s)

    # Check cache validity
    if [[ -f "$cache_file" ]]; then
        local cache_time
        cache_time=$(stat -f %m "$cache_file" 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null)
        if [[ $((now - cache_time)) -lt $_HETERODYNE_CACHE_TTL ]]; then
            cat "$cache_file"
            return
        fi
    fi

    # Refresh cache: find YAML files in current and config directories
    {
        find . -maxdepth 2 \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "config" ]] && find config \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "configs" ]] && find configs \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
    } | sort -u | tee "$cache_file"
}

# Get HDF5 data files
_heterodyne_get_data_files() {
    find . -maxdepth 3 \( -name "*.h5" -o -name "*.hdf5" -o -name "*.nxs" \) -type f 2>/dev/null
}

# Main heterodyne completion
_heterodyne() {
    local cur prev words cword
    _init_completion -s || return

    # Global options
    local global_opts="--config --data-file --method --verbose --quiet --help --version"

    # Method options
    local methods="nlsq cmc both"

    # Subcommands (if any)
    local subcommands=""

    case "$prev" in
        --config|-c)
            # Complete with YAML config files
            mapfile -t COMPREPLY < <(compgen -W "$(_heterodyne_get_config_files)" -- "${cur}")
            return
            ;;
        --data-file|-d)
            # Complete with HDF5 files
            mapfile -t COMPREPLY < <(compgen -W "$(_heterodyne_get_data_files)" -- "${cur}")
            return
            ;;
        --method|-m)
            # Complete with available methods
            mapfile -t COMPREPLY < <(compgen -W "${methods}" -- "${cur}")
            return
            ;;
        --output|-o)
            # Complete with directories
            _filedir -d
            return
            ;;
        --log-level)
            mapfile -t COMPREPLY < <(compgen -W "DEBUG INFO WARNING ERROR" -- "${cur}")
            return
            ;;
    esac

    # If current word starts with -, complete options
    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${global_opts}" -- "${cur}")
        return
    fi

    # Default: complete with config files
    mapfile -t COMPREPLY < <(compgen -W "$(_heterodyne_get_config_files) ${global_opts}" -- "${cur}")
}

# heterodyne-config completion
_heterodyne_config() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--output --template --minimal --verbose --help"

    case "$prev" in
        --output|-o)
            _filedir
            return
            ;;
        --template|-t)
            mapfile -t COMPREPLY < <(compgen -W "default minimal cmc" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# heterodyne-post-install completion
_heterodyne_post_install() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--interactive --shell --no-completion --no-xla --help"

    case "$prev" in
        --shell|-s)
            mapfile -t COMPREPLY < <(compgen -W "bash zsh fish" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# heterodyne-cleanup completion
_heterodyne_cleanup() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--dry-run --force --interactive --help"

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# heterodyne-validate completion
_heterodyne_validate() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--verbose --json --help"

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# Register completions
complete -F _heterodyne heterodyne
complete -F _heterodyne_config heterodyne-config
complete -F _heterodyne_post_install heterodyne-post-install
complete -F _heterodyne_cleanup heterodyne-cleanup
complete -F _heterodyne_validate heterodyne-validate

# Short aliases (ht = heterodyne)
complete -F _heterodyne ht
complete -F _heterodyne_config ht-config
complete -F _heterodyne_post_install ht-post-install
complete -F _heterodyne_cleanup ht-cleanup
complete -F _heterodyne_validate ht-validate

# Plotting aliases
alias hexp='heterodyne --plot-experimental-data'
alias hsim='heterodyne --plot-simulated-data'
complete -F _heterodyne hexp
complete -F _heterodyne hsim
