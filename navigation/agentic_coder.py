import os
from pickle import STACK_GLOBAL
import shutil
from typing_extensions import deprecated
from typing import Any, Optional
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks import UsageMetadataCallbackHandler
from datetime import datetime
from uuid import UUID
from na_utils.timer import MultiStartEndTimer

# New: reuse context generators to align tool intent to "context-only"
from gen_static import get_target_function_id
import gen_navigation
import gen_navigation_plus
from coder_eval_fit import get_project_root_dict as coder_eval_get_project_root_dict

class FileLogHandler(BaseCallbackHandler):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def _write_log(self, message):
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    def on_chain_start(self, serialized, inputs, **kwargs):
        self._write_log(f"[Chain Start] serialized:{serialized} \n Inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        self._write_log(f"[Chain End] Outputs: {outputs}")
    
    def on_tool_start(
        self,
        serialized,
        input_str: str,
        *,
        run_id,
        parent_run_id= None,
        tags=None,
        metadata = None,
        inputs= None,
        **kwargs,
    ):
        # self._write_log(f"[Tool Start] Input: {input_str}")
        self._write_log(f"[Tool Start] serialized: {serialized},\n Input: {input_str}")
        # print(f'on_tool_start: {input_str} ')

    # def on_tool_start(self, serialized, input_str, **kwargs):
    #     self._write_log(f"[Tool Start] Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        self._write_log(f"[Tool End] Output: {output}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._write_log(f"[LLM Start] Prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self._write_log(f"[LLM End] Response: {response}")

class AgentTimerCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.timer = MultiStartEndTimer()
        self.invoke_count = 0
        
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        self.invoke_count += 1
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        self.timer.start()
        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None, **kwargs: Any) -> Any:
        self.timer.end()
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)
    
    def get_elapsed_ms(self) -> float:
        return self.timer.get_elapsed_ms()

    def get_invoke_count(self) -> int:
        return self.invoke_count
    
    

class AgentHelper:
    STATES = {}
    CODER_EVAL_PROJECT_ROOT_DICT = None
    def __init__(self, meta_data, prompt_data, tmp_base_dir, task_id=0, dataset_name='DevEval') -> None:
        self.meta_data = meta_data
        self.task_id = task_id
        self.tmp_base_dir = tmp_base_dir
        self.project_path = os.path.join(tmp_base_dir, self.get_relative_project_path_when_generating())
        self._run_dir = os.getcwd()
        self.prompt_data = prompt_data
        self.dataset_name = dataset_name
        AgentHelper.STATES[task_id] = {
            'task_id': task_id,
            'project_path': self.project_path,
            'function_name': self.prompt_data['function_name'],
            'line': self.meta_data['signature_position'][0],
            'completion_path': self.get_completion_file_path(),
            'meta_data': self.meta_data,
        }
        
    def get_proj_relative_path(self):
        if self.dataset_name == 'CoderEval':
            if AgentHelper.CODER_EVAL_PROJECT_ROOT_DICT is None:
                AgentHelper.CODER_EVAL_PROJECT_ROOT_DICT = coder_eval_get_project_root_dict()
                # for k, v in AgentHelper.CODER_EVAL_PROJECT_ROOT_DICT.items():
                #     print(k, v)
                # exit()
            return AgentHelper.CODER_EVAL_PROJECT_ROOT_DICT[self.meta_data['namespace']]
        else:
            return self.meta_data['project_path']
        
    def get_file_log_handlerdler(self):
        d = os.path.join(self.tmp_base_dir, 'log', self.meta_data['namespace'])
        if not os.path.exists(d):
            os.makedirs(d)
        time_str= datetime.now().strftime("%m-%d-%H-%M-%S")
        log_file_path = os.path.join(d, f'{time_str}_{self.task_id}.log')
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w", encoding="utf-8") as f:
                pass
        return FileLogHandler(log_file_path)
    
    def prompt_dict(self):
        requirement = self.meta_data.get('requirement')
        if isinstance(requirement, dict):
            functionality = requirement.get('Functionality', '')
            arguments = requirement.get('Arguments', '')
            requirement_text = f"Functionality:\n{functionality}\n\nArguments:\n{arguments}"
        else:
            requirement_text = str(requirement)
        return {
            'task_id': self.task_id,
            # 'project_path': self.get_relative_project_path_when_generating(),
            'function_name': self.prompt_data['function_name'],
            'line': self.meta_data['signature_position'][0],
            'completion_path': self.get_completion_file_path(),
            'input_code': self.prompt_data['input_code'],
            'requirement': requirement_text,
        }
    
    @classmethod
    def get_prompt_system(cls):
        return '''You are a senior programmer acting as a Context Collector.
Your sole goal is to gather and summarize the most relevant and help in-repo source code context to help implement the target function — not to write or modify the target code.

Use the provided tools to comprehensively explore the repository:
- read_dir_tree to understand structure; read_file to fetch ranges; read_function/read_class to extract precise definitions; semantic_search to locate related files; find_definition/find_references for symbols.

Rules:
- Restrict to the current project. 
- Prefer what the target is likely to depend on: same-class methods, class properties (self.<attr> initialization), helpers in the same file, called functions, nearby utilities, module-level constants/types.
- Do NOT include the target function's own body.
- Avoid external libraries and standard library source. Be concise and avoid duplicates.

Output format (MUST FOLLOW EXACTLY):
- Plain text only, a sequence of annotated code blocks and a overall summary of all the useful information for generating the target code you got.
- For each snippet, a header then the raw code:
  # Source for: <module_or_path_or_symbol>
  <code>
  <explanation about why this snippet is useful to generate the target function based on the function requirement>
- No explanations, no markdown fences, no commentary, no JSON. Only the concatenated headers and code blocks.'''

    @classmethod
    def get_prompt_system_plus(cls):
        return '''You are a senior programmer acting as a Context Collector.
Your sole goal is to gather and summarize the most relevant and help in-repo source code context to help implement the target function — not to write or modify the target code.


Rules:
- The semantic search tool (semantic_search) is very useful, and it is recommended to start your search with semantic_search.
- When exploring the codebase, it is advised to transition from fuzzy to precise searching. For fuzzy searches, it is recommended to use the semantic search tool, while for precise searches, other tools are suggested.
- If you believe that using tools other than semantic search can help you obtain the desired context more quickly, you may use other tools.
- Restrict to the current project. 
- Prefer what the target is likely to depend on: same-class methods, class properties (self.<attr> initialization), helpers in the same file, called functions, nearby utilities, module-level constants/types.
- Do NOT include the target function's own body.
- Avoid external libraries and standard library source. Be concise and avoid duplicates.

Output format (MUST FOLLOW EXACTLY):
- Plain text only, a sequence of annotated code blocks and a overall summary of all the useful information for generating the target code you got.
- For each snippet, a header then the raw code:
  # Source for: <module_or_path_or_symbol>
  <code>
  <explanation about why this snippet is useful to generate the target function based on the function requirement>
- No explanations, no markdown fences, no commentary, no JSON. Only the concatenated headers and code blocks.'''

    @classmethod
    def get_prompt_user(cls):
        return '''There is a Python project in the current directory.
Your task: COLLECT CONTEXT for generating the function `{function_name}` in file `{completion_path}` at line: {line} based on the function signature and function requirement {requirement}.

You must NOT generate or modify code for the target function. Instead, use the tools to gather only the most relevant in-repo source code that would help implement it, excluding the target function body itself.

Guidance:
- Start from the target function's file. 
- Start from the same file and same class; collect related helpers, class properties, and sibling methods.
- Collect related helpers, class properties, and sibling methods. When you find a helper function, also find an example of how it is used elsewhere in the project.
- Analyze call relationships. Understand what functions the target might call, and what functions call the target.
- Expand to closely related modules if necessary (e.g., directly called utilities or constants).
- Avoid virtual environments, build artifacts, hidden folders, and third-party/vendor code.
- Remove duplicates; keep the overall context concise but complete.
- At least Invoke 3 times of tools to get the overview of the repo before generating the final result



Function signature and requirement (for reference):
```Python
{input_code}
```

Output: return ONLY the collected and summarized as annotated code blocks exactly in the specified format. No explanations.
Your task_id is {task_id}; include it when calling tools.'''
    
    def get_relative_project_path_when_generating(self):
        return self.meta_data['namespace'] + '_' + str(self.task_id)
    
    def get_completion_file_path(self):
        if self.dataset_name == 'CoderEval':
            # print(self.meta_data['completion_path'])
            # print(self.get_proj_relative_path())
            # exit()
            return self.meta_data['completion_path']
        else:
            return self.meta_data['completion_path'][len(self.get_proj_relative_path())+1:]
    
    def change_dir(self):
        os.chdir(self.project_path)
    
    def set_back_run_dir(self):
        os.chdir(self._run_dir)
    
    def preprocess(self, source_code_dir):
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            
        src_dir = os.path.join(source_code_dir, self.get_proj_relative_path())
        dst_dir = self.project_path
        try:
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        except Exception as e:
            pass
        completion_file_path = self.get_completion_file_path()
        exclude_dirs = {'venv', 'env', '.git', '__pycache__', 'myenv', '.vscode', '.idea', 'tests'}
        for d in exclude_dirs:
            if d+'/' in completion_file_path:
                continue
            if os.path.exists(os.path.join(dst_dir, d)):
                shutil.rmtree(os.path.join(dst_dir, d))
        
        self.change_dir()
        
        if self.dataset_name == 'DevEval':
            sos, eos = self.meta_data['body_position'][0]-1, self.meta_data['body_position'][1]
            with open(completion_file_path, 'r') as f:
                file_lines = f.readlines()
            indent = ''
            if self.meta_data['indent'] > 0:
                indent = ' ' * self.meta_data['indent']
            file_lines = file_lines[:sos] + ['\n', indent + 'pass', '\n'] + file_lines[eos:]
        elif self.dataset_name == 'CoderEval':
            function_start = self.meta_data['signature_position'][0] - 1
            function_end = self.meta_data['body_position'][1]
            with open(completion_file_path, 'r') as f:
                file_lines = f.readlines()
            signature_line = file_lines[function_start]
            if signature_line.startswith(' '):
                # ic = (len(signature_line) - len(signature_line.lstrip(' ')))
                # print('indent count:', ic)
                signature_indent = (len(signature_line) - len(signature_line.lstrip(' '))) * ' '
                pass_indent = signature_indent + ' ' * 4
            elif signature_line.startswith('\t'):
                signature_indent = (len(signature_line) - len(signature_line.lstrip('\t'))) * '\t'
                pass_indent = signature_indent + '\t'
            else:
                signature_indent = ''
                pass_indent = ' ' * 4
            file_lines = file_lines[:function_start] + \
                ['\n',signature_indent + self.meta_data['signature'],'\n', pass_indent +  'pass', '\n'] + file_lines[function_end:]
            
            # print(''.join(file_lines))
            # print(self.meta_data['signature'], self.get_proj_relative_path(), self.meta_data['completion_path'])
            # print(self.meta_data['namespace'], self.meta_data['signature_position'][0], self.meta_data['body_position'][1])
            # exit()
            
        with open(completion_file_path, 'w') as f:
            f.write(''.join(file_lines))
        self.set_back_run_dir()
        
    def post_process(self):
        self.set_back_run_dir()
        if os.path.exists(self.project_path):
            try:
                shutil.rmtree(self.project_path)
            except Exception as e:
                print(f'Failed to delete directory {self.project_path}. Reason: {e}')
        
        AgentHelper.STATES.pop(self.task_id)
                
    # ---- New: Context-only generation helpers ----
    def _resolve_paths_and_id(self, source_code_dir: str):
        """
        Resolve absolute project root, target file, and target function id from meta/prompt.
        """
        project_root = os.path.join(source_code_dir, self.get_proj_relative_path())
        target_file = os.path.join(source_code_dir, self.meta_data['completion_path'])
        target_function_id = get_target_function_id(
            self.meta_data['completion_path'],
            self.get_proj_relative_path(),
            self.meta_data['namespace']
        )
        return project_root, target_file, target_function_id

    def generate_navigation_context(self, source_code_dir: str, include_tree: bool = True) -> str:
        """
        Build context using the basic Navigation heuristic (no code generation).
        """
        project_root, target_file, target_function_id = self._resolve_paths_and_id(source_code_dir)
        return gen_navigation.generate_navigation_context(
            project_root=project_root,
            target_file_path=target_file,
            target_function_id=target_function_id,
            include_tree=include_tree,
        )

    def generate_navigation_plus_context(self, source_code_dir: str, bm25_top_k: int = 5) -> str:
        """
        Build context using Navigation-Plus (BM25-ranked related files; no code generation).
        """
        project_root, target_file, target_function_id = self._resolve_paths_and_id(source_code_dir)
        return gen_navigation_plus.generate_navigation_plus_context(
            project_root=project_root,
            target_file_path=target_file,
            target_function_id=target_function_id,
            bm25_top_k=bm25_top_k,
        )
    
    
# @deprecated
def make_prompt(prompt_element_file, meta_data_path, output_file):
    ''' Placeholder for prompt_element_file, as dynamic prompts are used and make_prompt is not needed.
    '''
    data = []
    import json
    with open(prompt_element_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    
    meta_data = {}
    with open(meta_data_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            meta_data[js['namespace']] = js
    
    template = open(f'/root/workspace/code/anonymousCodeContext/prompt/template/navigation.txt', 'r').read()
    for d in data:
        m = meta_data[d['namespace']]
        agent = AgentHelper(m,'', '/root/workspace/code/RUN/tmp')
        function_name = d['function_name']
        prompt = template.format(project_path=agent.get_relative_project_path_when_generating(), 
                                 function_name=function_name,
                                 line=m['signature_position'][0],
                                 completion_path=agent.get_completion_file_path(),
                                 input_code=d['input_code'])
        d['prompt'] = prompt
        d['function_name'] = function_name
        d['line'] = m['signature_position'][0]
        del d['contexts_above']
        del d['contexts_below']
        # del d['input_code']
        with open(output_file, 'a') as f:
            f.write(json.dumps(d) + '\n')


# def get_agent(llm):
#     create_tool_calling_agent
            
            
            
if __name__ == '__main__':
    # meta_data = {"namespace": "boltons.funcutils.FunctionBuilder.get_sig_str", "type": "method", "project_path": "Utilities/boltons", "completion_path": "Utilities/boltons/boltons/funcutils.py", "signature_position": [822, 822], "body_position": [828, 839], "depen     dency": {"intra_class": ["boltons.funcutils.FunctionBuilder.varkw", "boltons.funcutils.FunctionBuilder.annotations", "bolton     s.funcutils.FunctionBuilder.args", "boltons.funcutils.FunctionBuilder.varargs", "boltons.funcutils.FunctionBuilder.kwonlyarg     s"], "intra_file": ["boltons.funcutils.inspect_formatargspec"], "cross_file": []}, "requirement": {"Functionality": "This fu     nction returns the signature of a function as a string. The signature includes the function arguments and annotations if spe     cified.", "Arguments": ":param self: FunctionBuilder. An instance of the FunctionBuilder class.\n:param with_annotations: bo     ol. Whether to include annotations in the signature. Defaults to True.\n:return: str. The function signature as a string."},      "tests": ["tests/test_funcutils_fb.py::test_get_invocation_sig_str", "tests/test_funcutils_fb_py3.py::test_get_invocation_s     ig_str"], "indent": 12, "signature": "def get_sig_str(self, with_annotations=True):"}
    # agent = Agent(meta_data, '/root/workspace/code/RUN/tmp')
    # agent.preprocess('/root/workspace/code/DevEval/Source_Code')
    make_prompt('/root/workspace/code/anonymousCodeContext/prompt/navigation_deveval_prompt_elements.jsonl',
                '/root/workspace/code/DevEval/data_fixed2.jsonl',
                '/root/workspace/code/anonymousCodeContext/prompt/agent_prompt.jsonl')