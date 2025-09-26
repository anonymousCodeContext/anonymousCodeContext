# from openai import OpenAI
import os
from typing import List
from tqdm import tqdm
import time, json
import multiprocessing
import argparse
import traceback
from langchain_openai import ChatOpenAI
import ast
import asyncio
from glob import glob
from datetime import datetime
from navigation.agentic_coder import AgentHelper, FileLogHandler, AgentTimerCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import UsageMetadataCallbackHandler
from na_utils.timer import TimeCallbackHandler
from na_utils.async_llm_gen import TaskAsyncRunner, TaskExecutor
from langchain_community.callbacks.manager import get_openai_callback
from navigation_plus_python.tools import NavigationPlus
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--moda',default='greedy' ,type=str, required=True)
    parser.add_argument('--api_key_file', type=str, required=True)
    parser.add_argument('--max_concurrent', type=int, default=10, required=True)
    parser.add_argument('--sample_id_key', type=str, default='namespace', required=True)
    parser.add_argument('--min_count', type=int, default='-1', required=False) 
    
    parser.add_argument('--T', type=float)
    parser.add_argument('--top_p', type=float)
    parser.add_argument('--N', type=int)
    
    return parser.parse_args()

def load_api(path: str):
    api_keys = []
    with open(path, 'r') as f:
        for line in f:
            key = line.strip()
            api_keys.append(key)
    return api_keys

    
def create_llm(model_name, 
               key='', 
               open_key='',
               temperature=0, 
               max_tokens=None,
               max_retries=2,
               top_p=0.95,
               timeout=180,
               **kwargs):
    openai_api_key=key
    openai_api_base='your url'
    kwargs = {
        'openai_api_key': openai_api_key,
        'openai_api_base': openai_api_base, 
        'temperature': temperature,
        'max_tokens': max_tokens,
        'max_retries': max_retries,
        'top_p' : top_p,
        **kwargs,
    }
    if model_name == 'gpt-oss':
        assert open_key != None
        # print('----key: ',  open_key)
        kwargs['openai_api_key'] = open_key
        kwargs['openai_api_base'] = 'your url'
        return ChatOpenAI(
            model='openai/gpt-oss-120b',
            timeout=timeout,
            **kwargs
        )
    elif model_name == 'qwen3_coder':
        assert open_key != None
        # print('----key: ',  open_key)
        kwargs['openai_api_key'] = open_key
        kwargs['openai_api_base'] = 'your url'
        return ChatOpenAI(
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            timeout=timeout,
            **kwargs
        )
    elif model_name == 'llama':
        assert open_key != None
        # print('----key: ',  open_key)
        kwargs['openai_api_key'] = open_key
        kwargs['openai_api_base'] = 'your url'
        return ChatOpenAI(
            model='meta-llama/Llama-3.3-70B-Instruct',
            timeout=timeout,
            **kwargs
        )
    elif model_name == 'qwen3':
        assert open_key != None
        # print('----key: ',  open_key)
        kwargs['openai_api_key'] = open_key
        kwargs['openai_api_base'] = 'your url'
        return ChatOpenAI(
            model='Qwen/Qwen3-235B-A22B-Instruct-2507',
            timeout=timeout,
            **kwargs
        )
    elif model_name == "gpt" or model_name == 'gpt-4':
        return ChatOpenAI(
            model='gpt-4o-2024-11-20',
            timeout=timeout,
            **kwargs)
    elif model_name == 'gemini':
        return ChatOpenAI(
            model='gemini-2.5-pro',
            timeout=timeout,
            **kwargs
        )
    elif model_name == 'claude':
        return ChatOpenAI(
            model='claude-sonnet-4-20250514-thinking',
            timeout=timeout,
            **kwargs
        )
    # elif model_name == "deepseek":
    #     return ChatDeepSeek(**kwargs)
    elif model_name == 'deepseek': 
        return ChatOpenAI(
            model='deepseek-r1-250528',
            timeout=timeout,
            **kwargs
        )
    else:
        return ChatOpenAI(
            model=model_name,
            timeout=timeout,
            **kwargs
        )
        # return None

class TaskLLMGen_Agent(TaskExecutor):
    def __init__(self,
                 prompt_data_dict: dict, 
                 meta_data_dict: dict,
                 run_base_dir: str,
                 source_code_dir: str,
                 model_name, 
                 api_key, 
                 open_key,
                 temperature, 
                 max_tokens, 
                 top_p, 
                 max_retries):
        super().__init__()
        # self.prompt_data = prompt_data
        self.llm = create_llm(model_name, 
                     key=api_key, 
                     open_key=open_key,
                     temperature=temperature, 
                     max_tokens=max_tokens, 
                     top_p=top_p, 
                     max_retries=max_retries)
        from langchain_core.prompts import ChatPromptTemplate
        # from langchain.agents import create_tool_calling_agent
        from navigation.tools import get_all_tools
        prompt = ChatPromptTemplate(
            [
                ('system', AgentHelper.get_prompt_system()),
                ('user', AgentHelper.get_prompt_user()),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        # NOTE: Using Navigation-Plus tools
        # self.navigator = NavigationPlus(source_code_dir) # Initialize with the base source code dir
        # self.navigator.index() # Index the whole codebase
        
        # from navigation_plus_python.tools import get_all_tools
        tools = get_all_tools(self.navigator)
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        # log_dir = os.path.join(run_base_dir, 'logs')
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        self.meta_data_dict = meta_data_dict
        self.prompt_data_dict = prompt_data_dict
        self.run_base_dir = run_base_dir
        self.source_code_dir = source_code_dir
        
    async def do_task_func(self, sample_name, task_id):
        meta_data = self.meta_data_dict[sample_name]
        prompt_data = self.prompt_data_dict[sample_name]
        agent_helper = AgentHelper(meta_data, prompt_data, self.run_base_dir, task_id=task_id)
        
        # Update navigator with the specific project path for the current task
        self.navigator.project_root = agent_helper.project_path
        
        agent_helper.preprocess(self.source_code_dir)
        log_handler = agent_helper.get_file_log_handlerdler()
        usage_handler = UsageMetadataCallbackHandler()
        # print('prompt_dict:', agent_helper.prompt_dict())
        # print(sample_name,' invoke llm')
        time_handler = AgentTimerCallbackHandler()
        with get_openai_callback() as cb:
            msg = await self.executor.ainvoke(agent_helper.prompt_dict(),{'callbacks': [log_handler, usage_handler, time_handler]})
            token_cost = cb.total_tokens
            # print('token_cost:', token_cost)
            # print('cb:', cb)
        
        # print('result msg:', msg)
        # agent_helper.post_process()
        content = msg['output']
        prompt_data['llm_output'] = content
        if content is None or len(content) == 0:
            return False, None
        prompt_data['task_id'] = task_id
        prompt_data['usage'] = usage_handler.usage_metadata
        prompt_data['time_cost'] = time_handler.get_elapsed_ms()
        prompt_data['invoke_count'] = time_handler.get_invoke_count()
        agent_helper.set_back_run_dir()
        agent_helper.post_process()
        prompt_data['signature'] = meta_data['signature']
        return True, prompt_data
        


class TaskLLMGen(TaskExecutor):
    def __init__(self,
                 prompt_data: dict, 
                 model_name, 
                 api_key, 
                 open_key,
                 temperature, 
                 max_tokens, 
                 top_p, 
                 max_retries,
                 meta_data_dict):
        super().__init__()
        self.prompt_data = prompt_data
        self.meta_data_dict = meta_data_dict
        
        
        self.llm = create_llm(model_name, 
                     key=api_key, 
                     open_key=open_key,
                     temperature=temperature, 
                     max_tokens=max_tokens, 
                     top_p=top_p, 
                     max_retries=max_retries)

    async def do_task_func(self, sample_name, task_id):
        # print(f"Doing task: {sample_name}, {task_id}")
        prompt = self.prompt_data[sample_name]
        meta_data = self.meta_data_dict[sample_name]
        usage_handler = UsageMetadataCallbackHandler()
        time_handler = TimeCallbackHandler()
        # print('prompt:', prompt['prompt'])
        # print(sample_name, ' invoke llm')
        msg = await self.llm.ainvoke(prompt['prompt'], config={'callbacks': [usage_handler, time_handler]})
        # print('msg:', msg)
        # exit()
        content = msg.content
        if content is None or len(content) == 0:
            return False, None
        prompt['llm_output'] = content
        prompt['task_id'] = task_id
        
        prompt['usage'] = usage_handler.usage_metadata
        prompt['time_cost'] = time_handler.get_elapsed_ms()
        prompt['signature'] = meta_data['signature']
        return True, prompt


def run(prompt_file, 
        output_dir, 
        model, 
        moda, 
        api_key_file,
        open_key, 
        max_concurrent, 
        sample_id_key, 
        min_count,
        task='baseline',
        meta_data_path=None,
        source_code_path=None,
        run_base_dir=None):
    if moda == 'greedy':
        temperature = 0
        top_p = None
        gen_num_per_sample = 1
    elif moda == 'sampling':
        temperature = 0.4
        top_p = 0.95
        gen_num_per_sample = 5
        
    
    api_pool = load_api(api_key_file)
    api_key = api_pool[0] 
    
    all_samples = dict()
    min_count = int(min_count)
    with open(prompt_file, 'r') as f:
        if min_count < 0:
            for line in f:
                js = json.loads(line)
                # all_samples.append(js)
                all_samples[js[sample_id_key]] = js
        else:
            for i, line in enumerate(f):
                if i >= min_count:
                    break
                js = json.loads(line)
                # all_samples.append(js)
                all_samples[js[sample_id_key]] = js
    # exit()
    
    print('all_samples:', len(all_samples))
    
    meta_data_dict = dict()
    with open(meta_data_path, 'r') as f:
        if min_count < 0:
            for line in f:
                js = json.loads(line)
                # all_samples.append(js)
                meta_data_dict[js[sample_id_key]] = js
        else:
            all_md = utils.load_json_data_as_dict(meta_data_path)
            for k in all_samples:
                meta_data_dict[k] = all_md[k]
            
            # for i, line in enumerate(f):
            #     if i > min_count:
            #         break
            #     js = json.loads(line)
            #     # all_samples.append(js)
            #     meta_data_dict[js[sample_id_key]] = js
    
    # if task != 'agent':
    #     try:
    #         for p in all_samples:
    #             if 'ground_truth' in p and p['ground_truth'] in p['prompt']:
    #                 print('ground_truth in prompt:', p['namespace'])
    #                 exit()
    #             if 'signature' in p and p['signature'] in p['prompt']:
    #                 print('signature in prompt:', p['namespace'])
    #                 exit()
    #     except:
    #         pass
    llm_gen_executor = TaskLLMGen(
        prompt_data=all_samples,
        model_name=model, 
        api_key=api_key, 
        open_key=open_key,
        temperature=temperature, 
        max_tokens=None, 
        top_p=top_p, 
        max_retries=3,
        meta_data_dict=meta_data_dict
    )
    # else:
        # meta_data_dict: dict,
        #          run_base_dir: str,
        #          source_code_dir: str,
        
            
    # llm_gen_executor = TaskLLMGen_Agent(
    #         prompt_data_dict=all_samples,
    #         meta_data_dict=meta_data_dict,
    #         run_base_dir=run_base_dir,
    #         source_code_dir=source_code_path,
    #         model_name=model, 
    #         api_key=api_key, 
    #         open_key=open_key,
    #         temperature=temperature, 
    #         max_tokens=None, 
    #         top_p=top_p, 
    #         max_retries=1,
    #     )
    
    runner = TaskAsyncRunner(
            checkpoint_path=output_dir,
            taskExecutor=llm_gen_executor,
            all_samples=all_samples,
            sample_id_key=sample_id_key,
            gen_per_sample=gen_num_per_sample,
            max_concurrent_num=max_concurrent,
        )
    fail_count = runner.run()   
    print(f"Fail count: {fail_count}")
    return fail_count
        


def main():
    args = parse_args()
    
    if args.moda == 'greedy':
        args.T = 0
        args.top_p = None
        args.N = 1
        print('Using greedy mode')
    elif args.moda == 'sampling':
        args.T = 0.4
        args.top_p = 0.95
        args.N = 5
    print(args)  
    model_name = args.model

    api_pool = load_api(args.api_key_file)
    api_key = api_pool[0] 
    max_concurrent = args.max_concurrent
    sample_id_key = args.sample_id_key
    min_count = args.min_count
    
    run(args.prompt_file, 
        args.output_dir, 
        model_name, 
        args.moda, 
        api_key, 
        max_concurrent, 
        sample_id_key, 
        min_count)
    

if __name__ == "__main__":
    main()