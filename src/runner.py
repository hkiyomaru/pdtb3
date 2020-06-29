"""
http://www.apache.org/licenses/LICENSE-2.0
Usage: python scripts/runner.py <command-written file>
"""
import os
import sys
import subprocess as sp
import csv
import io
import shlex
import argparse
import copy
import asyncio

NVSMI = os.getenv('NVSMI_BIN', 'nvidia-smi')


class Task(object):
    def __init__(self, index, line, statdir):
        self.stderr = self.stdout = None
        self.process = self.waiter = None
        self.index = index
        self.parts = shlex.split(line)
        self.dirname = "%s/%05d" % (statdir, index)
        self.did_run = os.path.exists(self.dirname + "/stdout")
        self.out_fn = self.dirname + '/stdout'
        self.err_fn = self.dirname + '/stderr'
        self.gpu = None

    async def launch(self, env, gpu):
        os.makedirs(self.dirname, exist_ok=True)
        self.gpu = gpu

        self.stdout = open(self.out_fn, 'wb')
        self.stderr = open(self.err_fn, 'wb')
        self.process = await asyncio.create_subprocess_exec(
            *self.parts,
            stdout=self.stdout,
            stderr=self.stderr,
            env=env
        )
        self.waiter = self.process.wait()

    def line(self):
        return " ".join(self.parts)

    def close(self):
        self.stdout.close()
        self.stderr.close()


class GpuScheduler(object):
    def __init__(self, args):
        proc = sp.run([NVSMI, '--query-gpu=uuid', '--format=csv,noheader'], stdout=sp.PIPE)
        reader = csv.reader(io.StringIO(proc.stdout.decode('utf-8')))
        self.devices = set(x[0] for x in reader)
        self.max_running = args.max_gpus
        if self.max_running is None:
            self.max_running = len(self.devices)
        self.running = []

    async def run(self, t):
        gpu = await self.wait_for_gpu()
        env = copy.copy(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = gpu
        await t.launch(env, gpu)
        self.running.append(t)
        print(f"Started [{t.index}/{gpu}]: {t.line()}")

    def empty_gpus(self):
        proc = sp.run([NVSMI, '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], stdout=sp.PIPE)
        reader = csv.reader(io.StringIO(proc.stdout.decode('utf-8')))
        used_devices = set(x[0] for x in reader)
        for t in self.running:
            used_devices.add(t.gpu)

        return list(self.devices.difference(used_devices))

    async def wait_for_gpu(self):
        self.cleanup_processes()
        while True:
            available = self.empty_gpus()
            if len(available) > 0 and len(self.running) < self.max_running:
                return available[0]
            waits = [x.waiter for x in self.running]
            await asyncio.wait(waits, timeout=5, return_when=asyncio.FIRST_COMPLETED)
            self.cleanup_processes()

    def cleanup_processes(self):
        for t in self.running:
            if t.process.returncode is not None:
                self.running.remove(t)
                print(f"Finished [{t.index}]: {t.line()}")
                t.close()

                if t.process.returncode != 0 and os.path.exists(t.err_fn):
                    with open(t.err_fn, 'rt', encoding='utf-8') as f:
                        for line in f:
                            print(line.rstrip(), file=sys.stderr)

    async def wait_for_finish(self):
        self.cleanup_processes()
        waits = [x.waiter for x in self.running]
        while len(waits) > 0:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            self.cleanup_processes()
            waits = [x.waiter for x in self.running]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--max-gpus', type=int, default=None)
    p.add_argument('--used-with-memory', type=int, default=100*1024*1024)
    p.add_argument('launch')
    return p.parse_args()


if __name__ == '__main__':
    # Clean the variable for ourselves
    os.unsetenv('CUDA_VISIBLE_DEVICES')
    args = parse_args()
    sched = GpuScheduler(args)
    tasks = []

    statdir = args.launch + '.d'

    os.makedirs(statdir, exist_ok=True)

    with open(args.launch) as lines:
        for i, line in enumerate(lines):
            tasks.append(Task(i, line, statdir))

    loop = asyncio.get_event_loop()

    for t in tasks:
        if not t.did_run:
            loop.run_until_complete(sched.run(t))
    loop.run_until_complete(sched.wait_for_finish())
    loop.close()
