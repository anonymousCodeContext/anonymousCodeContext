import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
import uuid
import aiofiles

class Checkpoint:
    def __init__(self, ckpt_path):
        self.ckpt_path = Path(ckpt_path)
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        self.latest_file = self.ckpt_path / "latest.jsonl"
        self.initialized = False
        # self.current_ckpt_file = None

    def _get_new_ckpt_filename(self):
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return self.ckpt_path / f"ckpt-{date_str}.jsonl"

    def _get_latest_target(self):
        if self.latest_file.exists():
            return self.latest_file
        else:
            return None

    def read_latest(self):
        target = self._get_latest_target()
        if not target or not target.exists():
            return []
        with open(target, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    
        
    def create_to_do_task(self, all_samples:dict, sample_id_key='namespace', gen_per_sample=5):
        finished_task = self.read_latest()
        sample_dict = dict()
        for task in finished_task:
            k = task[sample_id_key]
            # print(k, sample_dict)
            if k not in sample_dict:
                sample_dict[k] = 0
            sample_dict[k] += 1
        to_do_task = []
        for key in all_samples:
            c = 0
            # key = s[sample_id_key]
            if key not in sample_dict:
                c = gen_per_sample
            else:
                c = gen_per_sample - sample_dict[key]
            if c > 0:
                to_do_task += [key for _ in range(c)] 
        return to_do_task


    async def append(self, data, id_key=None):
        if not self.initialized:
            latest_target = self._get_latest_target()
            new_ckpt_file = self._get_new_ckpt_filename()
            if latest_target and latest_target.exists():
                with open(latest_target, "r") as src, \
                     open(new_ckpt_file, "w") as dst:
                    for line in src:
                        dst.write(line)
            else:
                new_ckpt_file.touch()

            if self.latest_file.exists():
                self.latest_file.unlink()
            os.link(new_ckpt_file, self.latest_file)

            self.initialized = True
        if id_key is not None and id_key not in data:
            data[id_key] = str(uuid.uuid4())
        line = json.dumps(data, ensure_ascii=False) + "\n"
        async with aiofiles.open(self.latest_file, "a") as f:
            await f.write(line)
        

if __name__ == "__main__":
    import asyncio
    ckpt = Checkpoint("test_ckpt")
    async def test(i):
        await ckpt.append({"test": f"test{i}"})
        print(f"test{i}")
    
    async def main():
        await asyncio.gather(*[test(i) for i in range(10)])
    
    asyncio.run(main())