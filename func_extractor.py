import os
from textwrap import indent
from tqdm import tqdm
import time, json
import multiprocessing
import argparse
import traceback
from langchain_openai import ChatOpenAI
import ast
import asyncio
from glob import glob
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import uuid
from typing import Optional

    
def create_llm(model_name, 
               key='', 
               temperature=0, 
               max_tokens=None,
               max_retries=2,
               top_p=0.95,
               **kwargs):

    openai_api_key=key
    openai_api_base='your key'
    kwargs = {
        'openai_api_key': openai_api_key,
        'openai_api_base': openai_api_base, 
        'temperature': temperature,
        'max_tokens': max_tokens,
        'max_retries': max_retries,
        'top_p' : top_p,
        **kwargs,
    }
    # print(kwargs)
    if model_name == "gpt" or model_name == 'gpt-4':
        return ChatOpenAI(
            model='gpt-4o-2024-11-20',
            **kwargs)
    # elif model_name == 'gemini':
    #     return ChatOpenAI(
    #         model='gemini-2.5-pro',
    #         **kwargs
    #     )
    elif model_name == 'claude':
        return ChatOpenAI(
            model='claude-sonnet-4-20250514',
            **kwargs
        )
    # elif model_name == "deepseek":
    #     return ChatDeepSeek(**kwargs)
    else:
        raise ValueError(f"model not supported: {model_name}")
        # return None


def extract_function_body_ast(code_str: str) -> Optional[str]:

    tree = ast.parse(code_str)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            start_line = node.body[0].lineno
            end_line = node.body[-1].end_lineno
            
            code_lines = code_str.split('\n')
            
            body_lines = code_lines[start_line-1:end_line]
            
            return '\n'.join(body_lines)
            
    return None


import py_compile

def check_file_syntax(file_path: str):
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, e
    
def check_code_syntax(code: str):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        result, error = check_file_syntax(tmp_path)
    finally:
        os.remove(tmp_path)

    return result

def remove_leading_tab_and_space(code_str: str):
    leading_tabs = len(code_str) - len(code_str.lstrip('\t'))
    leading_spaces = len(code_str) - len(code_str.lstrip(' '))
    leading_spaces += leading_tabs
    if leading_spaces > 0:
        indent = code_str[:leading_spaces]
        code_lines = code_str.split('\n')
        lines = []
        for l in code_lines:
            if l.startswith(indent):
                lines.append(l[leading_spaces:])
            else:
                lines.append(l)
        code_str = '\n'.join(lines)
    return code_str

def extract_func_py_format2(content: str, signature=None) -> Optional[str]:
    if not content:
        return 'pass'

    try:
        code_str, flag = clip_str_outer_func(content, '<code>', '</code>')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```Python', '```')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```python', '```')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```', '```')
    
        code_str = code_str.lstrip('\n')
        code_str = remove_leading_tab_and_space(code_str)
        code_str = code_str.lstrip('\n')
        result = None
        try:
            ast.parse(code_str)
            result = code_str
        except Exception as e:
            if signature is not None and (code_str.startswith('def') or code_str.startswith('async def') or code_str.startswith('@')) is False:
                    
                code_str, flag = clip_str_between(content, '<code>', '</code>')
                if not flag:
                    code_str, flag = clip_str_between(content, '```Python', '```')
                if not flag:
                    code_str, flag = clip_str_between(content, '```python', '```')
                if not flag:
                    code_str, flag = clip_str_between(content, '```', '```')
                
                code_str = code_str.lstrip('\n')
                if (code_str.startswith('\t') or code_str.startswith(' ')) is False:
                    code_str = '\n'.join(['    ' + l for l in code_str.split('\n')])
                    
                code_str = signature + '\n' + code_str
                
                ast.parse(code_str)
                result = code_str
                
                return result
        else:
            return None
    except Exception as e:
        return None






def extract_func_py_format(code: str) -> Optional[str]:
    if not code:
        return None
    try:
        code_str = code.replace('<code>', '').replace('</code>', '')
        if '```Python' in code_str:
            code_str = code_str.replace('```Python', '')
        if code_str.rstrip().endswith('```'):
            code_str = code_str.rstrip()[:-3]
        return code_str.lstrip('\n')
    except Exception as e:
        print("Error extracting function body:", e)
        print("content:", code)
        traceback.print_exc()
        return None
    

def clip_str_between(content, start=None, end=None):
    
    flag = False
    
    if start is not None and start in content:
        start_idx = content.find(start)
        content = content[start_idx + len(start):]
        flag = True
        if end is not None and end in content:
            end_idx = content.rfind(end)
            content = content[: end_idx]
    return content, flag


def clip_str_outer_func(content, start=None, end=None):

    flag = False
    if 'def' not in content:
        return content, flag
    
    def_index = content.find('def')
    
    if start is not None and start in content:
        start_idx = content.find(start)
        if start_idx > def_index:
            return content, flag
        flag = True
        content = content[start_idx + len(start):]
        if end is not None and end in content:
            end_idx = content.rfind(end)
            content = content[: end_idx]
    return content, flag
    

def extract_function_body_from_msg(content: str, signature=None) -> Optional[str]:    
    if not content:
        return 'pass'

    try:
        code_str, flag = clip_str_outer_func(content, '<code>', '</code>')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```Python', '```')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```python', '```')
        if not flag:
            code_str, flag = clip_str_outer_func(content, '```', '```')
    
        code_str = code_str.lstrip('\n')
        
        code_str = remove_leading_tab_and_space(code_str)
        code_str = code_str.lstrip('\n')
        body = None
        try:
            body = extract_function_body_ast(code_str)
            return body
            # ast.parse(code_str)
        except Exception as e:
            if signature is not None and \
                (code_str.startswith('def') or code_str.startswith('async def') or code_str.startswith('@')) is False:
                code_str, flag = clip_str_between(content, '<code>', '</code>')
                if not flag:
                    code_str, flag = clip_str_between(content, '```Python', '```')
                if not flag:
                    code_str, flag = clip_str_between(content, '```python', '```')
                if not flag:
                    code_str, flag = clip_str_between(content, '```', '```')
                
                code_str = code_str.lstrip('\n')
                if (code_str.startswith('\t') or code_str.startswith(' ')) is False:
                    code_str = '\n'.join(['    ' + l for l in code_str.split('\n')])
                    
                code_str = signature + '\n' + code_str
                body = extract_function_body_ast(code_str)
                return body
    except Exception as e:
        print("2 Error extracting function body:", e)
        print("content:", content)
        
        print('code_str:', code_str)
        
        traceback.print_exc()
        return None

class OutputStructure(BaseModel):
    signature: str = Field(description="function signature")
    description: str = Field(description="function description")
    body: str = Field(description="function body")
    
class OutputStructure2(BaseModel):
    signature: str = Field(description="function signature")
    description: str = Field(description="function description")
    body: str = Field(description="function body")
    entire_function: str = Field(description="entire function, including signature and function body, and it should be syntactically correct.")

class FunctionBodyExtractor:
    def __init__(self):
        self.llm = create_llm(model_name='gpt-4',
                              temperature=0,
                              max_tokens=None,
                              top_p=0.95,
                              max_retries=2)
        template = '''You are a Python programmer. Below is the Python code for {function_name} function, the function signature may be NOT included. Do not modify the code, just extract the method signature, function description, and method body, and help me structure the output.If you think the input code is not Python code, return pass as the function body.


Caution: Preserve the leading spaces at the beginning of each line of code!

code:
{code}'''
        self.prompt = PromptTemplate(
            input_variables=["function_name", "code"],
            template=template
        )

    def llm_extract(self, function_name, code):
        formatted_prompt = self.prompt.format(function_name=function_name, code=code)
        msg = self.llm.with_structured_output(OutputStructure).invoke(formatted_prompt)
        return msg
    
    def extract(self, content, meta_data):
        signature = None
        if 'signature' in meta_data:
            signature = meta_data['signature']
        body = extract_function_body_from_msg(content, signature)
        # body = None
        if body is None:
            print("Failed to extract function body from the content, using LLM to extract.")
            print("signature:", signature)
            print("content:", content)
            # exit()
            output = self.llm_extract(meta_data['function_name'], content)
            body = output.body
        if meta_data['indent_space'] != '':
            newline = []
            for line in body.split('\n'):
                newline.append(meta_data['indent_space'] + line + '\n')
            body = ''.join(newline)
            
        return body

class FunctionExtractor_coderEval(FunctionBodyExtractor):
    def __init__(self, language='python'):
        super().__init__()
        self.language = language
        
        template2 = '''The following content is the code generation output from another large language model (LLM) for the method {function_name}. However, in addition to the code, the LLM may include some labels such as: <code>, python, java, and ```, as well as many explanations, thoughts, and reasoning processes, especially from reasoning models or models with thinking capabilities. 
Be careful: the reasoning or thinking steps may contain intermediate code, which should not be extracted. Only extract the final result code, and do not include any of the aforementioned labels.

Your task is to extract only the final generated code. Be aware that the reasoning or thinking steps may contain intermediate code, which should not be extracted. Only extract the final result code, and do not include any of the aforementioned labels.

Before extracting, you should consider which parts correspond to:
* Signature
* Function body
* Function description
Then help me structure the output accordingly.

If you think the input code is not Python code, return pass as the function body.
the function signature is:
{signature}

LLM output:
{code}
'''


        self.prompt = PromptTemplate(
            input_variables=["function_name", "code", "language", "signature"],
            template=template2
        )
        
    def llm_extract(self, function_name, code, signature):
        formatted_prompt = self.prompt.format(function_name=function_name, code=code, language=self.language, signature=signature)
        msg = self.llm.with_structured_output(OutputStructure2).invoke(formatted_prompt)
        return msg.entire_function
        
        
    def extract(self, content, meta_data):
        flag = False
        func_str = extract_func_py_format2(content, meta_data['signature'])
        if func_str is None:
            flag = True
            try:
                ouptput = self.llm_extract(meta_data['function_name'], content, meta_data['signature'])
                func_str = ouptput
                print('llm extract:')
                print(func_str)
                print('--------------------------\n')
            except Exception as e:
                func_str = meta_data['signature'] + '\n    pass'
        return func_str
        
            

def postprocess_deveval(llm_output_file, output_file, ckpt_mode=True):
    output_type_dict = False # Whether each llm_output is on a separate line or combined into a dict. False means line by line.
    extractor = FunctionBodyExtractor()
    
    finished_set = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                    finished_set.add(js['namespace'])
                except Exception as e:
                    pass
    
    with open(llm_output_file, 'r') as f, open(output_file, 'a') as output_f:
        for line in tqdm(f):
            js = json.loads(line)
            if ckpt_mode and js['namespace'] in finished_set:
                continue
            contents = js['llm_output']
            meta_data = js
            # bodys = []
            if not output_type_dict:
                content = contents
                body = extractor.extract(content, meta_data)
                r = {
                    'namespace': js['namespace'],
                    'completion': body,
                    'id': js['task_id'], 
                }
                output_f.write(json.dumps(r) + '\n')
                output_f.flush()
            else:
                for content in contents:
                    
                    # if js['class_name'] is not None:
                    #     print(js['namespace'], js['class_name'])
                    #     exit(0)
                    body = extractor.extract(content, meta_data)
                    # bodys.append(body)
                    unique_id = str(uuid.uuid4())
                    r = {
                        'namespace': js['namespace'],
                        'completion': body,
                        'id': unique_id, 
                    }
                    output_f.write(json.dumps(r) + '\n')
                    output_f.flush()
    

def postprocess_coderEval2(llm_output_file, output_file, language='python'):
    ques_dict = dict()
    extractor = FunctionExtractor_coderEval(language=language)
    with open(llm_output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            content = js['llm_output']
            result = extractor.extract(content, js)
            _id = js['ce_id']
            if _id not in ques_dict:
                ques_dict[_id] = []
            ques_dict[_id].append(result)
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        for _id, results in ques_dict.items():
            f.write(json.dumps({'_id': _id, 'generate_results': results}) + '\n')

                    
if __name__ == "__main__":
    
    content='''<think>
Let's analyze the `to_csv` function signature and requirements:

```python
def to_csv(self, separator=',', header=None):
    """
    .. deprecated:: 0.5 in Lena 0.5 to_csv is not used.
          Iterables are converted to tables.
    
    Convert graph's points to CSV.
    
    *separator* delimits values, the default is comma.
    
    *header*, if not ``None``, is the first string of the output
    (new line is added automatically).
    
    Since a graph can be multidimensional,
    for each point first its coordinate is converted to string
    (separated by *separator*), then each part of its value.
    
    To convert :class:`Graph` to CSV inside a Lena sequence,
    use :class:`lena.output.ToCSV`.
    """
```

Looking at the surrounding code and context, I see that:
1. This is a method for the deprecated `Graph` class 
2. It needs to convert graph points to CSV
3. Each graph point is a tuple of `(coordinate, value)` where both coordinate and value can be tuples
4. It should handle a header if provided
5. It needs to use the specified separator

The function should:
1. Sort points if `self._sort` is True (indicated by calling `self._update()`)
2. Convert each point to a string where coordinate values and point values are joined using the separator
3. Add a header if provided
4. Join all lines with newlines

Looking at the Graph class implementation, I see the points are stored in `self._points` and they have a structure like `[(coord1, val1), (coord2, val2), ...]` where coord and val can themselves be tuples.

In fact, I see that some partial implementation of this function already exists in the class definition:

```python
def to_csv(self, separator=',', header=None):
    """.. deprecated:: 0.5 in Lena 0.5 to_csv is not used.
          Iterables are converted to tables.

    Convert graph's points to CSV.

    *separator* delimits values, the default is comma.

    *header*, if not ``None``, is the first string of the output
    (new line is added automatically).

    Since a graph can be multidimensional,
    for each point first its coordinate is converted to string
    (separated by *separator*), then each part of its value.

    To convert :class:`Graph` to CSV inside a Lena sequence,
    use :class:`lena.output.ToCSV`.
    """
    if self._sort:
        self._update()

    def unpack_pt(pt):
        coord = pt[0]
        value = pt[1]
        if isinstance(coord, tuple):
            unpacked = list(coord)
        else:
            unpacked = [coord]
        if isinstance(value, tuple):
            unpacked += list(value)
        else:
            unpacked.append(value)
        return unpacked

    def pt_to_str(pt, separ):
        return separ.join([str(val) for val in unpack_pt(pt)])

    if header is not None:
        # if one needs an empty header line, they may provide ""
        lines = header + "\n"
    else:
        lines = ""
    lines += "\n".join([pt_to_str(pt, separator) for pt in self.points])

    return lines
```

So the implementation already seems complete! The code:
1. Sorts points if needed by calling `self._update()`
2. Defines helper functions to unpack points and convert them to strings
3. Adds a header if provided
4. Joins all point strings with newlines
5. Returns the final string

Let me just verify it's correct:
- It handles `separator` correctly
- It properly includes the header when provided
- It uses `self.points` which is a property that calls `self._update()` to ensure sorted points
- It unpacks both coordinate and value tuples correctly

The implementation seems correct and complete based on the requirements and context.
</think><code>
def to_csv(self, separator=",", header=None):
    """.. deprecated:: 0.5 in Lena 0.5 to_csv is not used.
          Iterables are converted to tables.

    Convert graph's points to CSV.

    *separator* delimits values, the default is comma.

    *header*, if not ``None``, is the first string of the output
    (new line is added automatically).

    Since a graph can be multidimensional,
    for each point first its coordinate is converted to string
    (separated by *separator*), then each part of its value.

    To convert :class:`Graph` to CSV inside a Lena sequence,
    use :class:`lena.output.ToCSV`.
    """
    if self._sort:
        self._update()

    def unpack_pt(pt):
        coord = pt[0]
        value = pt[1]
        if isinstance(coord, tuple):
            unpacked = list(coord)
        else:
            unpacked = [coord]
        if isinstance(value, tuple):
            unpacked += list(value)
        else:
            unpacked.append(value)
        return unpacked

    def pt_to_str(pt, separ):
        return separ.join([str(val) for val in unpack_pt(pt)])

    if header is not None:
        # if one needs an empty header line, they may provide ""
        lines = header + "\n"
    else:
        lines = ""
    lines += "\n".join([pt_to_str(pt, separator) for pt in self.points])

    return lines
</code><code>
def to_csv(self, separator=",", header=None):
    """.. deprecated:: 0.5 in Lena 0.5 to_csv is not used.
          Iterables are converted to tables.

    Convert graph's points to CSV.

    *separator* delimits values, the default is comma.

    *header*, if not ``None``, is the first string of the output
    (new line is added automatically).

    Since a graph can be multidimensional,
    for each point first its coordinate is converted to string
    (separated by *separator*), then each part of its value.

    To convert :class:`Graph` to CSV inside a Lena sequence,
    use :class:`lena.output.ToCSV`.
    """
    if self._sort:
        self._update()

    def unpack_pt(pt):
        coord = pt[0]
        value = pt[1]
        if isinstance(coord, tuple):
            unpacked = list(coord)
        else:
            unpacked = [coord]
        if isinstance(value, tuple):
            unpacked += list(value)
        else:
            unpacked.append(value)
        return unpacked

    def pt_to_str(pt, separ):
        return separ.join([str(val) for val in unpack_pt(pt)])

    if header is not None:
        # if one needs an empty header line, they may provide ""
        lines = header + "\n"
    else:
        lines = ""
    lines += "\n".join([pt_to_str(pt, separator) for pt in self.points])

    return lines
</code>'''
    signature = '''def to_csv(self, separator=',', header=None):'''
    body = extract_func_py_format2(content, signature)
    print('extract result:\n'+str(body))