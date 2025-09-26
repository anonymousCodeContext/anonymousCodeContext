from re import A
from na_utils.async_llm_gen import TaskAsyncRunner, TaskExecutor
from langChain_inference_async import create_llm
from navigation.agentic_coder import AgentHelper, FileLogHandler, AgentTimerCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import UsageMetadataCallbackHandler
import os
from navigation.tools2 import get_all_tools
import json
import traceback
from langchain_core.prompts import ChatPromptTemplate

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
                 max_retries,
                 dataset_name,
                 max_iterations=30,
                 plus=False):
        super().__init__()
        self.llm = create_llm(model_name, 
                     key=api_key, 
                     open_key=open_key,
                     temperature=temperature, 
                     max_tokens=max_tokens, 
                     top_p=top_p, 
                     max_retries=max_retries)
        self.dataset_name = dataset_name

        prompt = self.get_prompt()
        tools = self.get_tools()
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                      max_iterations=max_iterations)
        
        self.meta_data_dict = meta_data_dict
        self.prompt_data_dict = prompt_data_dict
        self.run_base_dir = run_base_dir
        self.source_code_dir = source_code_dir
    
    def get_tools(self):
        return get_all_tools()
    
    def get_prompt(self):
        return ChatPromptTemplate(
            [
                ('system', AgentHelper.get_prompt_system()),
                ('user', AgentHelper.get_prompt_user()),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
    async def do_task_func(self, sample_name, task_id):
        meta_data = self.meta_data_dict[sample_name]
        prompt_data = self.prompt_data_dict[sample_name]
        agent_helper = AgentHelper(meta_data, prompt_data, self.run_base_dir, task_id=task_id, dataset_name=self.dataset_name)
        usage_handler = UsageMetadataCallbackHandler()
        time_handler = AgentTimerCallbackHandler()
        content = ""
        try:
            agent_helper.preprocess(self.source_code_dir)
            log_handler = agent_helper.get_file_log_handlerdler()
            msg = await self.executor.ainvoke(
                agent_helper.prompt_dict(),
                {'callbacks': [log_handler, usage_handler, time_handler]}
            )
            if hasattr(msg, 'content'):
                content = msg.content
            if content is None or len(content) == 0:
                content = msg.get('output', '') if isinstance(msg, dict) else ''
            
        except Exception as e:
            print('error occur:', e)
            content = ""
            traceback.print_exc()
        finally:
            try:
                agent_helper.set_back_run_dir()
                agent_helper.post_process()
            except Exception:
                pass

        prompt_data['llm_output'] = content if content is not None else ""
        prompt_data['task_id'] = task_id
        try:
            prompt_data['usage'] = usage_handler.usage_metadata
        except Exception:
            prompt_data['usage'] = {}
        try:
            prompt_data['time_cost'] = time_handler.get_elapsed_ms()
            prompt_data['invoke_count'] = time_handler.get_invoke_count()
        except Exception:
            prompt_data['time_cost'] = 0
            prompt_data['invoke_count'] = 0
        prompt_data['signature'] = meta_data['signature']
        return True, prompt_data


class TaskLLMGen_Agent_Plus(TaskLLMGen_Agent):
    def get_tools(self):
        """
        Overrides the get_tools method of the parent class to return a custom list of tools.
        """
        from navigation_plus_python.tools3 import get_all_tools
        return get_all_tools()


def get_prompt_ctx(prompt_elem_file='./prompt/deveval_source_code2_prompt_elements.jsonl',
                meta_data_path='data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                run_base_dir='/root/workspace/code/RUN/tmp/agent/generate_prompt/running_env',
                source_code_path='/root/workspace/code/DevEval/Source_Code2',
                model='qwen3',
                max_iterations=30,
                api_key='',
                open_key='your key',
                min_count=-1,
                sample_id_key='namespace',
                output_path='/root/workspace/code/RUN/agent_generate_prompt/ckpt',
                dataset_name='DevEval',
                plus=False,
                max_concurrent_num=20,
                ):
    
    os.makedirs(output_path, exist_ok=True)
    all_samples = dict()
    with open(prompt_elem_file, 'r') as f:
        if min_count < 0:
            for line in f:
                js = json.loads(line)
                all_samples[js[sample_id_key]] = js
        else:
            for i, line in enumerate(f):
                if i >= min_count:
                    break
                js = json.loads(line)
                all_samples[js[sample_id_key]] = js
                
    meta_data_dict = dict()
    with open(meta_data_path, 'r') as f:
        if min_count < 0:
            for line in f:
                js = json.loads(line)
                meta_data_dict[js[sample_id_key]] = js
        else:
            for i, line in enumerate(f):
                if i > min_count:
                    break
                js = json.loads(line)
                meta_data_dict[js[sample_id_key]] = js
    
    print('all_samples:', len(all_samples))
    Agent_class = TaskLLMGen_Agent_Plus if plus else TaskLLMGen_Agent
    llm_gen_executor = Agent_class(
            prompt_data_dict=all_samples,
            meta_data_dict=meta_data_dict,
            run_base_dir=run_base_dir,
            source_code_dir=source_code_path,
            model_name=model, 
            api_key=api_key, 
            open_key=open_key,
            temperature=0, 
            max_tokens=None, 
            top_p=None, 
            max_retries=3,
            max_iterations=max_iterations,
            dataset_name=dataset_name,
        )
    
    runner = TaskAsyncRunner(
            checkpoint_path=output_path,
            taskExecutor=llm_gen_executor,
            all_samples=all_samples,
            sample_id_key=sample_id_key,
            gen_per_sample=1,
            max_concurrent_num=max_concurrent_num,
        )
    fail_count = runner.run()   
    print(f"Fail count: {fail_count}")
    

def make_prompt(prompt_ctx_file='/root/workspace/code/RUN/tmp/agent/generate_prompt/ckpt/qwen3/latest.jsonl',
                template_path='./prompt/template/navigation_template_refine.txt',
                elem_data_path='./prompt/deveval_source_code2_prompt_elements.jsonl',
                meta_data_path='data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                output_file='data/DevEval/navigation_prompt.jsonl',
                key='namespace'):
    import utils
    ctx = utils.load_json_data(prompt_ctx_file)
    elem_data = utils.load_json_data_as_dict(elem_data_path)
    meta_data = utils.load_json_data_as_dict(meta_data_path)
    
    template = open(template_path, 'r').read()
    with open(output_file, 'w') as f:
        for c in ctx:
            key_id = c[key]
            data = elem_data[key_id]
            md = meta_data[key_id]
            prompt = template.format(
                function_name=md['signature'],
                requirement=md['requirement'],
                relevant_codes=c.get('llm_output', ''),
                contexts_above=c.get('contexts_above', ''),
                contexts_below=c.get('contexts_below', ''),
                input_code=c.get('input_code', data.get('input_code', '')),
            )
            data['prompt'] = prompt
            f.write(json.dumps(data) + '\n')
    
if __name__ == '__main__':
    get_prompt_ctx()
    # make_prompt()