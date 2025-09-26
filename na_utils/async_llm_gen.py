import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from na_utils.checkpoint import Checkpoint

import asyncio
from typing import List
# from checkpoint import Checkpoint
import uuid

class ErrorCounter:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.count = 0
        self.lock = asyncio.Lock()

    async def increment(self):
        print("loop in ErrorCounter increment:", asyncio.get_running_loop())
        async with self.lock:
            self.count += 1
            return self.count

    async def reached(self):
        async with self.lock:
            return self.count >= self.tolerance

class TaskExecutor:
    def __init__(self):
        pass
    
    async def do_task_func(self, sample_name, task_id):

        return False, None

        
class TaskAsyncRunner:
    def __init__(self, 
                 checkpoint_path, 
                 taskExecutor: TaskExecutor,
                 all_samples: dict , 
                 sample_id_key='namespace',
                 gen_per_sample=5, 
                 max_concurrent_num=10,
                 err_tolerance=10,
                 ):

        self.ckpt = Checkpoint(ckpt_path=checkpoint_path)
        self.all_samples = all_samples
        self.max_concurrent_num = max_concurrent_num
        self.sem_max_concurrent = None
        self.gen_per_sample = gen_per_sample
        self.sample_id_key = sample_id_key
        self.taskExecutor = taskExecutor
        
    def _get_to_do_task(self):
        to_do_tasks = self.ckpt.create_to_do_task(self.all_samples,
                                    self.sample_id_key,
                                    self.gen_per_sample)
        
        print(f"Sample count: {len(self.all_samples)}, Total to do tasks: {len(to_do_tasks)}")
        return to_do_tasks
    
    async def _limited_task(self, sample_name, task_id, i=-1):
        # print("loop in _limited_task:", asyncio.get_running_loop())
        # print("sem_max_concurrent:", self.sem_max_concurrent._loop is asyncio.get_running_loop())
        async with self.sem_max_concurrent:
            try:
                flag, result = await self.taskExecutor.do_task_func(sample_name, task_id)
                if flag:
                    await self.ckpt.append(result)
                    print(f"Success task: {sample_name}, {task_id}, {i}")
                    return True
                else:
                    print(f"Failed task: {sample_name}, {task_id}, {i}")
                    return False
                # print(f"Success task: {sample_name}, {task_id}")
                # await append_checkpoint(namespace, idx, result, run_id)
                # return True
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error occur:", e)
                return False
            
    async def _async_run(self):
        # print("loop in _async_run:", asyncio.get_running_loop())
        to_do_tasks = self._get_to_do_task()
        if len(to_do_tasks) == 0:
            print('without to do task')
            return 0
        self.sem_max_concurrent = asyncio.Semaphore(self.max_concurrent_num)
        async_to_do_tasks = [self._limited_task(task, str(uuid.uuid4()), i) for i, task in enumerate(to_do_tasks)]
        print('start async task...')
        results = await asyncio.gather(*async_to_do_tasks)
        success_results = [r for r in results if r is not False]
        print(f"Total success this run: {len(success_results)}")
        return len(results) - len(success_results)
    
    def run(self):
        result =  asyncio.run(self._async_run())
        print('result:', result)
        return result

        
        
        
        
    
        