import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_EXCLUDES = [
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "artifacts_remote",
    "*.pyc",
    ".DS_Store",
]
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "NODE_FAIL"}
RUNTIME_PYTHON_PACKAGES = [
    "accelerate==1.13.0",
    "bert-score==0.3.13",
    "bitsandbytes==0.49.2",
    "datasets==4.8.4",
    "numpy==2.4.3",
    "peft==0.18.1",
    "scipy==1.17.1",
    "sentence-transformers==5.3.0",
    "transformers @ git+https://github.com/huggingface/transformers.git@main",
    "trl @ git+https://github.com/huggingface/trl.git@main",
]


@dataclass
class RemoteConfig:
    ssh_host: str
    ssh_user: str
    ssh_port: int
    ssh_key_path: Optional[Path]
    remote_root: Path
    local_root: Path
    hf_home: Path
    wandb_mode: str
    local_sync_root: Optional[Path] = None


@dataclass
class RemoteCommandSet:
    sync: str
    setup: str
    run: str
    sync_results: str = ""


def _shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def _ssh_prefix(config):
    parts = ["ssh"]
    if config.ssh_key_path:
        parts.extend(["-i", str(config.ssh_key_path)])
    parts.extend(["-p", str(config.ssh_port), f"{config.ssh_user}@{config.ssh_host}"])
    return _shell_join(parts)


def _ssh_bash_command(config, command):
    remote_command = f"bash -lc {shlex.quote(command)}"
    return f"{_ssh_prefix(config)} {shlex.quote(remote_command)}"


def _rsync_ssh_transport(config):
    transport = f"ssh -p {config.ssh_port}"
    if config.ssh_key_path:
        transport += f" -i {config.ssh_key_path}"
    return transport


def _scp_prefix(config):
    parts = ["scp", "-P", str(config.ssh_port)]
    if config.ssh_key_path:
        parts.extend(["-i", str(config.ssh_key_path)])
    return _shell_join(parts)


def _rsync_command(config):
    parts = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        _rsync_ssh_transport(config),
    ]
    for pattern in DEFAULT_EXCLUDES:
        parts.extend(["--exclude", pattern])
    parts.extend([f"{config.local_root}/", f"{config.ssh_user}@{config.ssh_host}:{config.remote_root}/"])
    return _shell_join(parts)


def _tar_sync_command(config):
    parts = ["tar", "-czf", "-"]
    for pattern in DEFAULT_EXCLUDES:
        parts.extend(["--exclude", pattern])
    parts.extend(["-C", str(config.local_root), "."])
    remote_extract = (
        f"mkdir -p {shlex.quote(str(config.remote_root))} && "
        f"tar --warning=no-unknown-keyword -xzf - -C {shlex.quote(str(config.remote_root))} && "
        f"find {shlex.quote(str(config.remote_root))} -name '._*' -type f -delete"
    )
    return (
        "COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 "
        f"{_shell_join(parts)} | {_ssh_bash_command(config, remote_extract)}"
    )


def _sync_command(config):
    rsync_command = _rsync_command(config)
    tar_command = _tar_sync_command(config)
    return (
        "if command -v rsync >/dev/null 2>&1 && "
        "! (rsync --version 2>/dev/null | grep -qi openrsync); then "
        f"{rsync_command}; "
        "else "
        f"{tar_command}; "
        "fi"
    )


def _local_sync_root(config, command_name):
    if config.local_sync_root:
        return Path(config.local_sync_root)
    family = "retrieval" if command_name.startswith("retrieval") else "generation"
    return config.local_root / "artifacts_remote" / family / config.remote_root.name


def _remote_setup_body(config):
    return (
        f"mkdir -p {shlex.quote(str(config.remote_root))} "
        f"{shlex.quote(str(config.remote_root / 'logs'))} "
        f"{shlex.quote(str(config.remote_root / '.cache' / 'huggingface'))} "
        f"{shlex.quote(str(config.remote_root / '.cache' / 'triton'))} "
        f"{shlex.quote(str(config.remote_root / '.cache' / 'pip'))} "
        f"{shlex.quote(str(config.remote_root / '.bootstrap'))}"
    )


def _remote_python_command(config, module_name, args):
    env_prefix = f"HF_HOME={shlex.quote(str(config.hf_home))} WANDB_MODE={shlex.quote(config.wandb_mode)}"
    return (
        f"cd {shlex.quote(str(config.remote_root))} && "
        ". .venv/bin/activate && "
        f"{env_prefix} python -m {module_name} {' '.join(args)}"
    )


def _slurm_script_name(command_name):
    return f"{command_name.replace('-', '_')}.sbatch"


def _shared_python_job_setup_lines(env_slug):
    package_args = " ".join(shlex.quote(package) for package in RUNTIME_PYTHON_PACKAGES)
    return [
        'mkdir -p "$WORKDIR" "$WORKDIR/logs" "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/triton" "$WORKDIR/.cache/pip" "$WORKDIR/.bootstrap"',
        'export HF_HOME="$WORKDIR/.cache/huggingface"',
        'export HF_DATASETS_CACHE="$HF_HOME/datasets"',
        'export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"',
        'export TRANSFORMERS_CACHE="$HF_HOME/transformers"',
        'export TRITON_CACHE_DIR="$WORKDIR/.cache/triton"',
        'export WANDB_DIR="$WORKDIR/wandb"',
        'export PIP_CACHE_DIR="$WORKDIR/.cache/pip"',
        'export XDG_CONFIG_HOME="$WORKDIR/.bootstrap/xdg"',
        'export CONDARC="$WORKDIR/.bootstrap/condarc"',
        'export CONDA_PKGS_DIRS="$WORKDIR/.bootstrap/conda-pkgs"',
        'export CONDA_ENVS_PATH="$WORKDIR/.bootstrap/conda-envs"',
        'export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"',
        'export PYTHONUNBUFFERED=1',
        "",
        'JOB_ENV_ROOT="$WORKDIR/.bootstrap/job_envs"',
        f'JOB_ENV="$JOB_ENV_ROOT/{env_slug}"',
        'ENV_READY_STAMP="$JOB_ENV/.elsst-env-ready-v1"',
        'MAMBA_ROOT="$WORKDIR/.bootstrap/micromamba-root"',
        'MAMBA_BIN="$WORKDIR/.bootstrap/bin/micromamba"',
        'ORIG_HOME="${HOME:-$WORKDIR}"',
        'mkdir -p "$JOB_ENV_ROOT" "$MAMBA_ROOT" "$WORKDIR/.bootstrap/bin" "$XDG_CONFIG_HOME" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"',
        "",
        'if [ -n "${HF_TOKEN:-}" ]; then',
        '  export HF_TOKEN="${HF_TOKEN}"',
        "fi",
        "",
        'if [ -f "$ORIG_HOME/.bashrc" ]; then',
        '  source "$ORIG_HOME/.bashrc"',
        "fi",
        "",
        'export HOME="$WORKDIR/.bootstrap/home"',
        'mkdir -p "$HOME"',
        'cat > "$CONDARC" <<EOF',
        'pkgs_dirs:',
        '  - $CONDA_PKGS_DIRS',
        'envs_dirs:',
        '  - $CONDA_ENVS_PATH',
        'always_yes: true',
        'EOF',
        "",
        'cd "$WORKDIR"',
        "",
        'if [ ! -x "$MAMBA_BIN" ]; then',
        '  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORKDIR/.bootstrap" bin/micromamba',
        "fi",
        "",
        'export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"',
        '"$MAMBA_BIN" --version',
        'if [ ! -d "$JOB_ENV" ]; then',
        '  "$MAMBA_BIN" create -y -p "$JOB_ENV" -c conda-forge python=3.11 pip git',
        "fi",
        '"$MAMBA_BIN" run -p "$JOB_ENV" python --version',
        'if [ ! -f "$ENV_READY_STAMP" ]; then',
        '  "$MAMBA_BIN" run -p "$JOB_ENV" python -m pip install --upgrade pip setuptools wheel',
        '  "$MAMBA_BIN" run -p "$JOB_ENV" python -m pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple torch==2.6.0',
        '  "$MAMBA_BIN" run -p "$JOB_ENV" python -m pip install -e . --no-deps',
        f'  "$MAMBA_BIN" run -p "$JOB_ENV" python -m pip install {package_args}',
        '  touch "$ENV_READY_STAMP"',
        "fi",
        '"$MAMBA_BIN" run -p "$JOB_ENV" python - <<\'PY\'',
        "import sys",
        "import torch",
        "print('torch', torch.__version__)",
        "print('cuda_available', torch.cuda.is_available())",
        "print('device_count', torch.cuda.device_count())",
        "if not torch.cuda.is_available():",
        "    raise SystemExit('CUDA unavailable in job environment')",
        "PY",
    ]


def _retrieval_job_spec(command_name):
    if command_name == "retrieval-sanity":
        return {
            "job_name": "elsst-ret-sanity-l40s",
            "output_dir": "artifacts/retrieval/sanity_l40s",
            "time_limit": "02:00:00",
            "extra_args": ["--preset", "full_stable", "--max-steps", "20", "--merge-adapter", "--resume-from-checkpoint", "auto"],
        }
    if command_name == "retrieval-full":
        return {
            "job_name": "elsst-ret-full-l40s",
            "output_dir": "artifacts/retrieval/full_l40s",
            "time_limit": "72:00:00",
            "extra_args": ["--preset", "full_stable", "--merge-adapter", "--resume-from-checkpoint", "auto"],
        }
    raise ValueError(f"unsupported retrieval slurm command: {command_name}")


def _render_retrieval_slurm_script(config, command_name):
    spec = _retrieval_job_spec(command_name)
    workdir = str(config.remote_root)
    job_name = spec["job_name"]
    output_dir = spec["output_dir"]
    entry_args = [
        "--dataset-root", "track1",
        "--output-dir", output_dir,
        "--model-name", "Qwen/Qwen3-Embedding-0.6B",
        *spec["extra_args"],
    ]
    joined_args = _shell_join(entry_args)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --partition=l40s_risk",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=8",
            f"#SBATCH --time={spec['time_limit']}",
            "#SBATCH --mem=64000M",
            "#SBATCH --output=logs/%x-%j.out",
            "#SBATCH --error=logs/%x-%j.err",
            "#SBATCH --gpus-per-node=1",
            "",
            "set -euo pipefail",
            f'WORKDIR="{workdir}"',
            *_shared_python_job_setup_lines("retrieval-cu124-py311"),
            f'"$MAMBA_BIN" run -p "$JOB_ENV" python -m elsst_baselines.retrieval.train {joined_args}',
            "",
        ]
    )


def _generation_job_spec(command_name):
    if command_name == "sft-smoke":
        return {
            "job_name": "elsst-sft-smoke-l40s",
            "module_name": "elsst_baselines.generation.train_sft",
            "output_dir": "artifacts/generation/sft_smoke_l40s",
            "time_limit": "04:00:00",
            "extra_args": [
                "--preset",
                "smoke",
                "--max-train-samples",
                "16",
                "--max-eval-samples",
                "16",
                "--max-steps",
                "2",
                "--resume-from-checkpoint",
                "auto",
            ],
        }
    if command_name == "sft-full":
        return {
            "job_name": "elsst-sft-full-l40s",
            "module_name": "elsst_baselines.generation.train_sft",
            "output_dir": "artifacts/generation/sft_full_l40s",
            "time_limit": "24:00:00",
            "extra_args": [
                "--preset",
                "auto",
                "--resume-from-checkpoint",
                "auto",
            ],
        }
    if command_name == "dpo-smoke":
        return {
            "job_name": "elsst-dpo-smoke-l40s",
            "module_name": "elsst_baselines.generation.train_dpo",
            "output_dir": "artifacts/generation/dpo_smoke_l40s",
            "time_limit": "04:00:00",
            "extra_args": [
                "--preset",
                "smoke",
                "--sft-adapter-dir",
                "artifacts/generation/sft_smoke_l40s/adapter",
                "--max-train-samples",
                "16",
                "--max-eval-samples",
                "16",
                "--max-steps",
                "2",
                "--resume-from-checkpoint",
                "auto",
            ],
        }
    if command_name == "dpo-full":
        return {
            "job_name": "elsst-dpo-full-l40s",
            "module_name": "elsst_baselines.generation.train_dpo",
            "output_dir": "artifacts/generation/dpo_full_l40s",
            "time_limit": "24:00:00",
            "extra_args": [
                "--preset",
                "auto",
                "--sft-adapter-dir",
                "artifacts/generation/sft_full_l40s/adapter",
                "--resume-from-checkpoint",
                "auto",
            ],
        }
    if command_name == "orpo-smoke":
        return {
            "job_name": "elsst-orpo-smoke-l40s",
            "module_name": "elsst_baselines.generation.train_orpo",
            "output_dir": "artifacts/generation/smoke_l40s",
            "time_limit": "04:00:00",
            "extra_args": ["--preset", "smoke", "--max-train-samples", "16", "--max-eval-samples", "16", "--max-steps", "2"],
        }
    if command_name == "orpo-full":
        return {
            "job_name": "elsst-orpo-full-l40s",
            "module_name": "elsst_baselines.generation.train_orpo",
            "output_dir": "artifacts/generation/full_l40s",
            "time_limit": "24:00:00",
            "extra_args": ["--preset", "auto"],
        }
    raise ValueError(f"unsupported generation slurm command: {command_name}")


def _render_generation_slurm_script(config, command_name):
    spec = _generation_job_spec(command_name)
    workdir = str(config.remote_root)
    job_name = spec["job_name"]
    output_dir = spec["output_dir"]
    module_name = spec["module_name"]
    entry_args = [
        "--dataset-root", "track2",
        "--output-dir", output_dir,
        "--model-name", "Qwen/Qwen3.5-4B",
        *spec["extra_args"],
    ]
    joined_args = _shell_join(entry_args)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --partition=l40s_risk",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=8",
            f"#SBATCH --time={spec['time_limit']}",
            "#SBATCH --mem=64000M",
            "#SBATCH --output=logs/%x-%j.out",
            "#SBATCH --error=logs/%x-%j.err",
            "#SBATCH --gpus-per-node=1",
            "#SBATCH --constraint=fs_weka",
            "",
            "set -euo pipefail",
            f'WORKDIR="{workdir}"',
            *_shared_python_job_setup_lines("generation-cu124-py311"),
            f'"$MAMBA_BIN" run -p "$JOB_ENV" python -m {module_name} {joined_args}',
            "",
        ]
    )


def _remote_write_and_submit_command(config, command_name, script_content):
    script_name = _slurm_script_name(command_name)
    encoded_script = base64.b64encode(script_content.encode("utf-8")).decode("ascii")
    write_command = (
        f": {shlex.quote(script_content)} && "
        f"cd {shlex.quote(str(config.remote_root))} && "
        "python3 - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        f"Path({script_name!r}).write_bytes(base64.b64decode({encoded_script!r}))\n"
        "PY\n"
        f"sbatch {shlex.quote(script_name)}"
    )
    return _ssh_bash_command(config, write_command)


def _sync_results_command(config, command_name):
    local_sync_root = _local_sync_root(config, command_name)
    scp_prefix = _scp_prefix(config)
    remote_prefix = f"{config.ssh_user}@{config.ssh_host}:{config.remote_root}"
    family = "retrieval" if command_name.startswith("retrieval") else "generation"
    parts = [
        f"mkdir -p {shlex.quote(str(local_sync_root / 'artifacts' / family))}",
        f"mkdir -p {shlex.quote(str(local_sync_root / 'logs'))}",
        f"mkdir -p {shlex.quote(str(local_sync_root / 'job_scripts'))}",
        (
            "{scp} -r {remote}/artifacts/{family}/. "
            "{local}/artifacts/{family}/ || true"
        ).format(
            scp=scp_prefix,
            remote=shlex.quote(remote_prefix),
            family=family,
            local=shlex.quote(str(local_sync_root)),
        ),
        (
            "{scp} -r {remote}/logs/. "
            "{local}/logs/ || true"
        ).format(
            scp=scp_prefix,
            remote=shlex.quote(remote_prefix),
            local=shlex.quote(str(local_sync_root)),
        ),
        (
            "{scp} '{remote}/*.sbatch' "
            "{local}/job_scripts/ || true"
        ).format(
            scp=scp_prefix,
            remote=remote_prefix,
            local=shlex.quote(str(local_sync_root)),
        ),
    ]
    return " && ".join(parts)


def build_remote_commands(config, command_name):
    sync_command = _sync_command(config)
    setup_command = _ssh_bash_command(config, _remote_setup_body(config))
    sync_results_command = ""

    if command_name == "retrieval-smoke":
        run_body = _remote_python_command(
            config,
            "elsst_baselines.retrieval.train",
            [
                "--dataset-root", "track1",
                "--output-dir", "artifacts/retrieval/smoke",
                "--model-name", "Qwen/Qwen3-Embedding-0.6B",
                "--preset", "smoke",
                "--max-train-samples", "64",
                "--max-eval-samples", "64",
                "--max-steps", "5",
            ],
        )
        run_command = _ssh_bash_command(config, run_body)
    elif command_name in {"retrieval-sanity", "retrieval-full"}:
        run_command = _remote_write_and_submit_command(
            config,
            command_name,
            _render_retrieval_slurm_script(config, command_name),
        )
        sync_results_command = _sync_results_command(config, command_name)
    elif command_name in {"sft-smoke", "sft-full", "dpo-smoke", "dpo-full", "orpo-smoke", "orpo-full"}:
        run_command = _remote_write_and_submit_command(
            config,
            command_name,
            _render_generation_slurm_script(config, command_name),
        )
        sync_results_command = _sync_results_command(config, command_name)
    elif command_name == "eval-all":
        retrieval_eval = (
            "cd {root} && . .venv/bin/activate && HF_HOME={hf_home} WANDB_MODE={wandb_mode} "
            "python -m elsst_baselines.retrieval.evaluate --dataset-root track1 --output-dir artifacts/retrieval/full "
            "--model-name Qwen/Qwen3-Embedding-0.6B --adapter-dir artifacts/retrieval/full/adapter"
        ).format(
            root=shlex.quote(str(config.remote_root)),
            hf_home=shlex.quote(str(config.hf_home)),
            wandb_mode=shlex.quote(config.wandb_mode),
        )
        generation_eval = (
            "cd {root} && . .venv/bin/activate && HF_HOME={hf_home} WANDB_MODE={wandb_mode} "
            "python -m elsst_baselines.generation.evaluate --dataset-root track2 --output-dir artifacts/generation/dpo_full_l40s "
            "--model-name Qwen/Qwen3.5-4B --adapter-dir artifacts/generation/dpo_full_l40s/adapter"
        ).format(
            root=shlex.quote(str(config.remote_root)),
            hf_home=shlex.quote(str(config.hf_home)),
            wandb_mode=shlex.quote(config.wandb_mode),
        )
        run_command = _ssh_bash_command(config, retrieval_eval + " && " + generation_eval)
    else:
        raise ValueError(f"unsupported command: {command_name}")

    return RemoteCommandSet(
        sync=sync_command,
        setup=setup_command,
        run=run_command,
        sync_results=sync_results_command,
    )


def _required_value(value, env_name):
    if value:
        return value
    raise SystemExit(f"{env_name} is required")


def _default_remote_root(ssh_user, command_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"elsst-{command_name}-{timestamp}"
    return f"/mnt/fast/nobackup/users/{ssh_user}/codex-surrey-runs/{run_name}"


def _config_from_args(args):
    ssh_host = _required_value(args.ssh_host or os.getenv("SSH_HOST"), "SSH_HOST")
    ssh_user = _required_value(args.ssh_user or os.getenv("SSH_USER"), "SSH_USER")
    remote_root = args.remote_root or os.getenv("REMOTE_ROOT") or _default_remote_root(ssh_user, args.command)
    ssh_port = int(args.ssh_port or os.getenv("SSH_PORT") or 22)
    ssh_key_path = args.ssh_key_path or os.getenv("SSH_KEY_PATH")
    hf_home = args.hf_home or os.getenv("HF_HOME") or f"{remote_root}/.cache/huggingface"
    wandb_mode = args.wandb_mode or os.getenv("WANDB_MODE") or "disabled"
    local_root = Path(args.local_root).resolve()
    local_sync_root = args.local_sync_root or os.getenv("LOCAL_SYNC_ROOT")
    return RemoteConfig(
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        ssh_key_path=Path(ssh_key_path) if ssh_key_path else None,
        remote_root=Path(remote_root),
        local_root=local_root,
        hf_home=Path(hf_home),
        wandb_mode=wandb_mode,
        local_sync_root=Path(local_sync_root).resolve() if local_sync_root else None,
    )


def _run_command(command, dry_run, capture_output=False):
    if dry_run:
        print(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return subprocess.run(
        command,
        shell=True,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _run_checked(command):
    result = _run_command(command, dry_run=False, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def _parse_job_id(stdout):
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise SystemExit("could not parse Slurm job id from sbatch output")
    return match.group(1)


def _watch_slurm_job(config, job_id, poll_seconds=30):
    while True:
        squeue_command = _ssh_bash_command(
            config,
            f"squeue -j {job_id} -o '%.18i %.9P %.25j %.8T %.10M %.6D %R' | tail -n +2",
        )
        squeue_result = _run_command(squeue_command, dry_run=False, capture_output=True)
        if squeue_result.returncode == 0 and squeue_result.stdout.strip():
            print(squeue_result.stdout.strip())
            time.sleep(poll_seconds)
            continue

        sacct_command = _ssh_bash_command(
            config,
            f"sacct -j {job_id} --format=JobID,State,Elapsed -P -n",
        )
        sacct_result = _run_command(sacct_command, dry_run=False, capture_output=True)
        if sacct_result.returncode != 0:
            time.sleep(poll_seconds)
            continue

        lines = [line.strip() for line in sacct_result.stdout.splitlines() if line.strip()]
        for line in lines:
            parts = line.split("|")
            if not parts or parts[0] != job_id:
                continue
            state = parts[1]
            elapsed = parts[2] if len(parts) > 2 else ""
            print(f"{job_id} {state} {elapsed}".strip())
            if state in TERMINAL_STATES:
                return state
        time.sleep(poll_seconds)


def _tail_remote_logs(config):
    command = _ssh_bash_command(
        config,
        (
            f"cd {shlex.quote(str(config.remote_root))} && "
            "for path in $(ls -1t logs/* 2>/dev/null | head -n 4); do "
            "echo \"===== ${path} =====\"; tail -n 80 \"$path\"; done"
        ),
    )
    _run_command(command, dry_run=False, capture_output=False)


def _run_direct_pipeline(config, command_name, dry_run):
    commands = build_remote_commands(config, command_name)
    if dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "command": command_name,
                    "remote_root": str(config.remote_root),
                    "sync": commands.sync,
                    "setup": commands.setup,
                    "run": commands.run,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for command in (commands.sync, commands.setup, commands.run):
        result = _run_command(command, False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _run_slurm_pipeline(config, command_name, dry_run):
    commands = build_remote_commands(config, command_name)
    local_sync_root = _local_sync_root(config, command_name)
    if dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "command": command_name,
                    "remote_root": str(config.remote_root),
                    "local_sync_root": str(local_sync_root),
                    "sync": commands.sync,
                    "setup": commands.setup,
                    "run": commands.run,
                    "sync_results": commands.sync_results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _run_checked(commands.sync)
    _run_checked(commands.setup)
    submit_result = _run_checked(commands.run)
    job_id = _parse_job_id(submit_result.stdout)
    state = _watch_slurm_job(config, job_id)
    _tail_remote_logs(config)
    if commands.sync_results:
        sync_result = _run_command(commands.sync_results, False)
        if sync_result.returncode != 0:
            return sync_result.returncode
    return 0 if state == "COMPLETED" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Remote GPU runner for ELSST baselines.")
    parser.add_argument(
        "command",
        choices=[
            "sync",
            "setup",
            "retrieval-smoke",
            "retrieval-sanity",
            "retrieval-full",
            "sft-smoke",
            "sft-full",
            "dpo-smoke",
            "dpo-full",
            "orpo-smoke",
            "orpo-full",
            "eval-all",
        ],
    )
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-port")
    parser.add_argument("--ssh-key-path")
    parser.add_argument("--remote-root")
    parser.add_argument("--local-root", default=Path(__file__).resolve().parents[3])
    parser.add_argument("--local-sync-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        config = _config_from_args(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "sync":
        return _run_command(build_remote_commands(config, "retrieval-smoke").sync, args.dry_run).returncode
    if args.command == "setup":
        return _run_command(build_remote_commands(config, "retrieval-smoke").setup, args.dry_run).returncode
    if args.command in {
        "retrieval-sanity",
        "retrieval-full",
        "sft-smoke",
        "sft-full",
        "dpo-smoke",
        "dpo-full",
        "orpo-smoke",
        "orpo-full",
    }:
        return _run_slurm_pipeline(config, args.command, args.dry_run)
    return _run_direct_pipeline(config, args.command, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
