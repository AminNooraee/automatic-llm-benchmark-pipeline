#!/bin/sh
set -eu

usage() {
  echo "Usage: sh scripts/run_benchmark.sh CONFIG [--project1-handoff FILE] [--skip-preflight]" >&2
  exit 64
}

[ "$#" -ge 1 ] || usage
config_input=$1
shift
handoff_input=
skip_preflight=0
while [ "$#" -gt 0 ]; do
  case $1 in
    --project1-handoff)
      [ "$#" -ge 2 ] || usage
      handoff_input=$2
      shift 2
      ;;
    --skip-preflight)
      skip_preflight=1
      shift
      ;;
    *) usage ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
[ -f "$config_input" ] || { echo "Configuration is not a regular file: $config_input" >&2; exit 66; }
config_dir=$(CDPATH= cd -- "$(dirname -- "$config_input")" && pwd -P)
config_path=$config_dir/$(basename -- "$config_input")
case $config_path in
  "$repo_dir"/*) ;;
  *) echo "Configuration must be inside the repository workspace: $repo_dir" >&2; exit 65 ;;
esac
config_relative=${config_path#"$repo_dir"/}

image=${BENCHMARK_IMAGE:-automatic-llm-benchmark-pipeline:local}
runs_input=${BENCHMARK_RUNS_DIR:-$repo_dir/runs}
mkdir -p -- "$runs_input"
runs_dir=$(CDPATH= cd -- "$runs_input" && pwd -P)

# Ensure the nested Docker bind target exists before /workspace is mounted read-only.
mkdir -p -- "$repo_dir/runs"

dataset_mount=
if [ -n "${BENCHMARK_DATASETS_DIR:-}" ]; then
  [ -d "$BENCHMARK_DATASETS_DIR" ] || { echo "BENCHMARK_DATASETS_DIR is not a directory" >&2; exit 66; }
  dataset_mount=$(CDPATH= cd -- "$BENCHMARK_DATASETS_DIR" && pwd -P)
fi

handoff_path=
if [ -n "$handoff_input" ]; then
  [ -f "$handoff_input" ] || { echo "Project #1 handoff is not a regular file: $handoff_input" >&2; exit 66; }
  handoff_dir=$(CDPATH= cd -- "$(dirname -- "$handoff_input")" && pwd -P)
  handoff_path=$handoff_dir/$(basename -- "$handoff_input")
fi

env_names=$(grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' "$config_path" 2>/dev/null | sed 's/^${//;s/}$//' | sort -u)
for variable_name in $env_names; do
  printf '%s\n' "$variable_name" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$' || {
    echo "Invalid environment reference name in configuration" >&2
    exit 65
  }
  printenv "$variable_name" >/dev/null 2>&1 || {
    echo "Required environment variable is not set: $variable_name" >&2
    exit 78
  }
done

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --tag "$image" "$repo_dir"
fi

set -- docker run --rm --user "$(id -u):$(id -g)" --read-only --tmpfs /tmp:rw,noexec,nosuid \
  --mount "type=bind,src=$repo_dir,dst=/workspace,readonly" \
  --mount "type=bind,src=$runs_dir,dst=/workspace/runs" \
  --workdir /workspace

if [ -n "$dataset_mount" ]; then
  set -- "$@" --mount "type=bind,src=$dataset_mount,dst=/datasets,readonly"
fi
if [ -n "$handoff_path" ]; then
  set -- "$@" --mount "type=bind,src=$handoff_path,dst=/handoff/project1.json,readonly"
fi
if [ -n "${BENCHMARK_DOCKER_NETWORK:-}" ]; then
  set -- "$@" --network "$BENCHMARK_DOCKER_NETWORK"
fi
for variable_name in $env_names; do
  set -- "$@" -e "$variable_name"
done
set -- "$@" "$image" run --config "/workspace/$config_relative"
if [ -n "$handoff_path" ]; then
  set -- "$@" --project1-handoff /handoff/project1.json
fi
if [ "$skip_preflight" -eq 1 ]; then
  set -- "$@" --skip-preflight
fi

echo "Benchmark reports will be written under: $runs_dir"
"$@"
