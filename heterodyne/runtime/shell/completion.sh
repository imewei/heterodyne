#!/bin/bash
# shellcheck disable=SC2034  # words/cword set by _init_completion convention
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

# Main heterodyne completion
_heterodyne() {
    local cur prev words cword
    _init_completion -s || return

    # All options grouped by category
    local global_opts="--config --method --output --output-format --verbose --quiet --help --version"
    local analysis_opts="--phi --multistart --multistart-n --num-samples --num-chains --cmc-backend --nlsq-result --no-nlsq-warmstart"
    local param_opts="--initial-D0-ref --initial-alpha-ref --initial-D-offset-ref --initial-D0-sample --initial-alpha-sample --initial-D-offset-sample --initial-v0 --initial-beta --initial-v-offset --initial-f0 --initial-phi0"
    local perf_opts="--threads --no-jit"
    local plot_opts="--plot --no-plot --plot-experimental-data --plot-simulated-data --contrast --offset-sim --save-plots --plotting-backend --parallel-plots --phi-angles"
    local all_opts="${global_opts} ${analysis_opts} ${param_opts} ${perf_opts} ${plot_opts}"

    case "$prev" in
        --config|-c)
            # Complete with YAML config files
            mapfile -t COMPREPLY < <(compgen -W "$(_heterodyne_get_config_files)" -- "${cur}")
            return
            ;;
        --method|-m)
            mapfile -t COMPREPLY < <(compgen -W "nlsq cmc both" -- "${cur}")
            return
            ;;
        --output|-o)
            _filedir -d
            return
            ;;
        --output-format)
            mapfile -t COMPREPLY < <(compgen -W "json npz both" -- "${cur}")
            return
            ;;
        --plotting-backend)
            mapfile -t COMPREPLY < <(compgen -W "auto matplotlib datashader" -- "${cur}")
            return
            ;;
        --threads|--multistart-n|--num-samples|--num-chains)
            # Integer arguments — no completion
            return
            ;;
        --cmc-backend)
            mapfile -t COMPREPLY < <(compgen -W "auto cpu multiprocessing pjit pbs" -- "${cur}")
            return
            ;;
        --nlsq-result)
            _filedir -d
            return
            ;;
        --phi|--contrast|--offset-sim|--initial-D0-ref|--initial-alpha-ref|--initial-D-offset-ref|--initial-D0-sample|--initial-alpha-sample|--initial-D-offset-sample|--initial-v0|--initial-beta|--initial-v-offset|--initial-f0|--initial-phi0)
            # Float arguments — no completion
            return
            ;;
    esac

    # If current word starts with -, complete options
    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${all_opts}" -- "${cur}")
        return
    fi

    # Default: complete with config files
    mapfile -t COMPREPLY < <(compgen -W "$(_heterodyne_get_config_files) ${all_opts}" -- "${cur}")
}

# heterodyne-config completion
_heterodyne_config() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--output --data --q --dt --time-length --overwrite --show-template --interactive --validate --mode --help"

    case "$prev" in
        --output|-o|--data|-d)
            _filedir
            return
            ;;
        --mode)
            mapfile -t COMPREPLY < <(compgen -W "full minimal nlsq_only cmc_only" -- "${cur}")
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

    local opts="--interactive --shell --no-completion --no-xla --xla-mode --verbose --help"

    case "$prev" in
        --shell|-s)
            mapfile -t COMPREPLY < <(compgen -W "bash zsh fish" -- "${cur}")
            return
            ;;
        --xla-mode)
            mapfile -t COMPREPLY < <(compgen -W "auto nlsq cmc cmc-hpc" -- "${cur}")
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

    local opts="--dry-run --force --interactive --verbose --help"

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

# heterodyne-config-xla completion
_heterodyne_config_xla() {
    # shellcheck disable=SC2034  # words/cword used by _init_completion
    local cur prev words cword
    _init_completion -s || return

    local opts="--threads --no-x64 --debug --info --help"

    case "$prev" in
        --threads)
            # Suggest common thread counts
            local cpu_count
            cpu_count=$(nproc 2>/dev/null || echo 4)
            mapfile -t COMPREPLY < <(compgen -W "1 2 4 8 ${cpu_count}" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# Register completions
complete -F _heterodyne heterodyne
complete -F _heterodyne_config heterodyne-config
complete -F _heterodyne_config_xla heterodyne-config-xla
complete -F _heterodyne_post_install heterodyne-post-install
complete -F _heterodyne_cleanup heterodyne-cleanup
complete -F _heterodyne_validate heterodyne-validate

# Short aliases (ht = heterodyne)
complete -F _heterodyne ht
complete -F _heterodyne ht-nlsq
complete -F _heterodyne ht-cmc
complete -F _heterodyne_config ht-config
complete -F _heterodyne_config_xla ht-xla
complete -F _heterodyne_post_install ht-setup
complete -F _heterodyne_cleanup ht-clean
complete -F _heterodyne_validate ht-validate

# Plotting aliases
complete -F _heterodyne hexp
complete -F _heterodyne hsim

# Shell aliases
alias ht='heterodyne'
alias ht-config='heterodyne-config'
alias ht-nlsq='heterodyne --method nlsq'
alias ht-cmc='heterodyne --method cmc'
alias hexp='heterodyne --plot-experimental-data'
alias hsim='heterodyne --plot-simulated-data'
alias ht-xla='heterodyne-config-xla'
alias ht-setup='heterodyne-post-install'
alias ht-clean='heterodyne-cleanup'
