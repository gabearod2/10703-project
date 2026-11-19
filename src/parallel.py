"""Run independent trials in separate processes.

`spawn` (not fork) is used so every worker gets its own clean CUDA context. Workers share the GPU; with
threads=1 each one is limited by a single CPU core, so use roughly one job per free core.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_parallel(fn, tasks, jobs, on_result):
    """Call fn(*task) for every task. `on_result(position, result)` runs in the parent as each finishes, where
    `result` is fn's return value or the exception that escaped it (e.g. a worker killed by the OOM killer).
    jobs <= 1 runs in this process, in order, exactly like a plain loop."""
    tasks = list(tasks)
    if jobs <= 1 or len(tasks) <= 1:
        for pos, task in enumerate(tasks):
            try:
                result = fn(*task)
            except Exception as error:
                result = error
            on_result(pos, result)
        return
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(fn, *task): pos for pos, task in enumerate(tasks)}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = error
            on_result(futures[future], result)
