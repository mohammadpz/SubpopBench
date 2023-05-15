import subprocess
import time
import torch
import sys
import getpass
from pathlib import Path


def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None] * n_gpus
    import ipdb; ipdb.set_trace()

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def slurm_launcher(commands, output_dirs, max_slurm_jobs=12):
    for output_dir, cmd in zip(output_dirs, commands):
        block_until_running(max_slurm_jobs, getpass.getuser())
        out_str = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
        print(out_str.strip())
        if output_dir:
            try:
                job_id = int(out_str.split(' ')[-1])
            except (IndexError, ValueError, AttributeError):
                print("Error in Slurm submission, exiting.")
                sys.exit(0)

            (Path(output_dir)/'job_id').write_text(str(job_id))


def submitit_launcher(args, args_list):
    import submitit
    import argparse
    from subpopbench.train import run_experiment, get_default_parser
    default_args = vars(get_default_parser().parse_args([]))

    # updating the default_args using args in the args_list
    for i, args_ in enumerate(args_list):
        args_copy = default_args.copy()
        args_copy.update(args_)
        args_list[i] = argparse.Namespace(**args_copy)

    executor = submitit.SlurmExecutor(folder=args.slurm_dir)
    executor.update_parameters(
        time=24 * 60,
        gpus_per_node=1,
        array_parallelism=512,
        cpus_per_task=4,
        constraint='volta32gb',
        partition=args.slurm_partition)
    executor.map_array(run_experiment, args_list)


def get_slurm_jobs(user):
    out = subprocess.run(['squeue -u ' + user], shell=True, stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
    a = list(filter(lambda x: len(x) > 0, map(lambda x: x.split(), out.split('\n'))))
    queued, running = [], []
    for i in a:
        if i[0].isnumeric():
            if i[4].strip() == 'PD':
                queued.append(int(i[0]))
            else:
                running.append(int(i[0]))
    return queued, running


def block_until_running(n, user):
    while True:
        queued, running = get_slurm_jobs(user)
        if len(queued) + len(running) < n:
            time.sleep(0.2)
            return True
        else:
            time.sleep(10)   


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm': slurm_launcher,
    'submitit': submitit_launcher
}
