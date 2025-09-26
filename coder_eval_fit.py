from argparse import Namespace
from time import perf_counter
from token import RPAR

from pydantic.type_adapter import P
import utils
import json
import os
import ast
from pathlib import Path


source_coder_eval_meta_data_path = '/root/workspace/code/CoderEval/docker_mount_out_data/python/CoderEval4Python_fixed.json'

proj2path = {
        'ufo-kit/concert' : 'ufo-kit---concert' ,
        'sunpy/radiospectra' : 'sunpy---radiospectra' ,
        'pexip/os-zope' : 'pexip---os-zope' ,
        'burgerbecky/makeprojects' : 'burgerbecky---makeprojects' ,
        'zimeon/ocfl-py' : 'zimeon---ocfl-py' ,
        'eykd/prestoplot' : 'eykd---prestoplot' ,
        'pexip/os-python-cachetools' : 'pexip---os-python-cachetools' ,
        'champax/pysolbase' : 'champax---pysolbase' ,
        'pexip/os-python-dateutil' : 'pexip---os-python-dateutil' ,
        'redhat-openstack/infrared' : 'redhat-openstack---infrared' ,
        'SoftwareHeritage/swh-lister' : 'SoftwareHeritage---swh-lister' ,
        'mozilla/relman-auto-nag' : 'mozilla---relman-auto-nag' ,
        'commandline/flashbake' : 'commandline---flashbake' ,
        'cpburnz/python-sql-parameters' : 'cpburnz---python-sql-parameters' ,
        'santoshphilip/eppy' : 'santoshphilip---eppy' ,
        'scrolltech/apphelpers' : 'scrolltech---apphelpers' ,
        'bazaar-projects/docopt-ng' : 'bazaar-projects---docopt-ng' ,
        'pre-commit/pre-commit' : 'pre-commit---pre-commit' ,
        'ynikitenko/lena' : 'ynikitenko---lena' ,
        'sipwise/repoapi' : 'sipwise---repoapi' ,
        'mwatts15/rdflib' : 'mwatts15---rdflib' ,
        'openstack/cinder' : 'openstack---cinder' ,
        'openstack/neutron-lib' : 'openstack---neutron-lib' ,
        'ossobv/planb' : 'ossobv---planb' ,
        'ikus060/rdiffweb' : 'ikus060---rdiffweb' ,
        'bastikr/boolean' : 'bastikr---boolean' ,
        'witten/borgmatic' : 'witten---borgmatic' ,
        'kirankotari/shconfparser' : 'kirankotari---shconfparser' ,
        'jaywink/federation' : 'jaywink---federation' ,
        'scieloorg/packtools' : 'scieloorg---packtools' ,
        'infobloxopen/infoblox-client' : 'infobloxopen---infoblox-client' ,
        'rougier/matplotlib' : 'rougier---matplotlib' ,
        'MozillaSecurity/lithium' : 'MozillaSecurity---lithium' ,
        'SEED-platform/py-seed' : 'SEED-platform---py-seed' ,
        'witten/atticmatic' : 'witten---atticmatic' ,
        'rak-n-rok/Krake' : 'rak-n-rok---Krake' ,
        'turicas/rows' : 'turicas---rows' ,
        'cloudmesh/cloudmesh-common' : 'cloudmesh---cloudmesh-common' ,
        'awsteiner/o2sclpy' : 'awsteiner---o2sclpy' ,
        'ansible-security/ansible_collections.ibm.qradar' : 'ansible-security---ansible_collections.ibm.qradar' ,
        'skorokithakis/shortuuid' : 'skorokithakis---shortuuid' ,
        'neo4j/neo4j-python-driver' : 'neo4j---neo4j-python-driver' ,
        'gopad/gopad-python': 'gopad',        
}

def read_source_meta_data():
    with open(source_coder_eval_meta_data_path, 'r') as f:
        content = f.read()
    data = json.loads(content)
    result = []
    for l in data['RECORDS']:
        result.append(l)
    return result
    
def find_class_name_of_method(code: str, method_name: str):
    """
    Given a code string and a method name, find the class name to which the method belongs.
    Returns None if the method does not belong to any class.
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        print("Error parsing code:", e)
        return None

    class ClassFinder(ast.NodeVisitor):
        def __init__(self, target_method):
            self.target_method = target_method
            self.found_class = None

        def visit_ClassDef(self, node):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == self.target_method:
                    self.found_class = node.name
                    return
                if isinstance(item, ast.AsyncFunctionDef) and item.name == self.target_method:
                    self.found_class = node.name
                    return
            for item in node.body:
                if isinstance(item, ast.ClassDef):
                    self.visit_ClassDef(item)

    finder = ClassFinder(method_name)
    finder.visit(tree)
    return finder.found_class

def get_above_below_by_code(code: str, target_method_name: str, start_line: int=0, end_line: int=999999):
    tree = ast.parse(code)
    code_lines = code.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
                temp_method_name = node.name
                if target_method_name != temp_method_name and "_"+target_method_name!=temp_method_name:
                    continue
                if node.lineno < start_line or node.end_lineno > end_line:
                    continue
                start_line = node.lineno
                end_line = node.end_lineno
                above = '\n'.join(code_lines[:start_line - 1])
                below = '\n '.join(code_lines[end_line + 1:]) if end_line is not None else ''
                node.body = [ast.Pass()]
                signature = ast.unparse(node)
                if signature:
                    signature = signature.replace('pass', '').rstrip()  
                return above, below, signature            
    return None, None, ''


def get_function_position_by_name_and_lines(code: str, function_name: str, start_line: int, end_line: int):
    """
    Get the position information of a function's signature and body based on its name and start/end line numbers.
    
    Args:
        code: The source code string.
        function_name: The name of the target function.
        start_line: The starting line number of the function in the file (1-based).
        end_line: The ending line number of the function in the file (1-based).
        
    Returns:
        tuple: (signature_start, signature_end, body_start, body_end)
               signature_start: The starting line number of the function signature (1-based).
               signature_end: The ending line number of the function signature (1-based).
               body_start: The starting line number of the function body (1-based).
               body_end: The ending line number of the function body (1-based).
    """
    try:
        tree = ast.parse(code)
        code_lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name and node.lineno >= start_line and node.end_lineno <= end_line:
                    signature_start = node.lineno
                    
                    signature_end = signature_start
                    for i in range(signature_start - 1, end_line):
                        line = code_lines[i].rstrip()
                        if line.endswith(':'):
                            signature_end = i + 1
                            break
                    
                    body_start = signature_end + 1
                    for i in range(signature_end, end_line):
                        line = code_lines[i].strip()
                        if line and not line.startswith('#'):
                            body_start = i + 1
                            break
                    
                    body_end = node.end_lineno
                    
                    return signature_start, signature_end, body_start, body_end
                    
        return None, None, None, None
        
    except Exception as e:
        print(f"Error parsing code: {e}")
        return None, None, None, None


def gen_prompt_elements4coder_eval_python():
    base_dir = '/root/workspace/code/CoderEval/docker_mount_out_data/python'
    with open(os.path.join(base_dir, 'CoderEval4Python.json'), 'r') as f:
        content = f.read()
    data = json.loads(content)
    result = []
    for l in data['RECORDS']:
        function_name = l['name']
        class_name = find_class_name_of_method(l['file_content'], function_name)
        above, below, signature = get_above_below_by_code(l['file_content'], function_name)
        if above is None or below is None or signature == '':
            print(f"Method {function_name} above or below is None")
        signature = signature.replace('pass', '').rstrip()        
        input_code = f'''{signature}
"""
{l['docstring']}
"""
'''
        e = {
                "class_name" : class_name, 
                "function_name": function_name,
                "contexts_above": above,
                "contexts_below": below,
                "input_code": input_code,
                "coderEval_id": l['_id'],
            }
        result.append(e)
    with open('./prompt/coderEval/navigation_prompt_elements.jsonl', 'w') as fout:
        for item in result:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_meta_data_4_coderEval(outpath = 'data/CoderEval/metadata_fixed.jsonl',
                                 elem_out_path = 'data/CoderEval/elements.jsonl'):
    
    source_meta_data = read_source_meta_data()
    fc = 0
    namespace_set = set()
    proj_root_dict = get_project_root_dict()
    with open(outpath, 'w') as fout, open(elem_out_path, 'w') as felem:
        for m in source_meta_data:
            s_file_path = m['file_path']
            s_function_name = m['name']
            namespace = s_file_path[:-3].replace('/', '.') + '.' + s_function_name
            namespace_set.add(namespace)
            s_class_name = find_class_name_of_method(m['file_content'], s_function_name)
            type = 'function'
            if s_class_name is not None:
                type = 'class'
            project_path = m['project']
            completion_path = s_file_path
            code_lines = m['file_content'].split('\n')
            start_line_file = int(m['lineno'])
            end_line_file = int(m['end_lineno'])
            sign_start, sign_end, body_start, body_end = \
                get_function_position_by_name_and_lines(m['file_content'], s_function_name, 
                                                    start_line_file, end_line_file)
            signature_position = [start_line_file, None]
            body_position = [None, end_line_file]
            above, below, signature = get_above_below_by_code(m['file_content'], s_function_name, start_line_file, end_line_file)
            ground_truth = '\n'.join(code_lines[start_line_file - 1: end_line_file])
            class_name = None
            if type != 'function':
                class_name = namespace.split('.')[-2]
            input_code = f'''{signature}
"""
{m['docstring']}
"""
'''
            space_num = len(code_lines[start_line_file]) - len(code_lines[start_line_file].lstrip(' '))
            real_proj_path = proj_root_dict[m['_id']]
            t_meta = {
                "namespace": m['_id'],
                "namespace_real": namespace,
                "type": type,
                "project_path": project_path,
                'real_proj_path': real_proj_path,
                "completion_path": completion_path,
                'ce_id': m['_id'],
                'signature_position': signature_position,
                'body_position': body_position,
                'signature': signature,
                'ground_truth': ground_truth,
                'requirement': {
                    'Functionality': m['docstring'],
                    'Arguments': '',
                }
            }
            
            e = {
                "ce_id": m['_id'],
                "namespace": m['_id'],
                "namespace_real": namespace,
                'real_proj_path': real_proj_path,
                "type": type,
                "class_name" : class_name, 
                "function_name": s_function_name,
                "contexts_above": above,
                "contexts_below": below,
                "input_code": input_code,
                "indent_space": ' ' * space_num,
                "ground_truth": ground_truth,
                "signature": signature,
                "completion_path": completion_path
            }
            fout.write(json.dumps(t_meta) + '\n')
            felem.write(json.dumps(e) + '\n')
    print('not pass count:', fc)
        
        
    
def output_fit():
    '''
    The generation result required by coderEval is very simple:
    {
        '_id': '',
        'generate_results':[
            "" # This is the answer, including the signature
        ]
    }
    '''
    pass

def create_dummy_source_code(output_dir='data/CoderEval/dummy_source_code'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    meta_data = read_source_meta_data()
    for m in meta_data:
        file_path = m['file_path']
        file_path = os.path.join(output_dir, file_path)
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(m['file_content'])
         
         

def copy_content_to_source_code_dir(source_code_dir='/root/workspace/code/CoderEval/docker_mount_out_data/python/repos'):
    meta_data = read_source_meta_data()
    meta_data_ori_dict = { m['_id']: m for m in meta_data }
    meta_data_na_dict = utils.load_json_data_as_dict('data/CoderEval/metadata_fixed.jsonl')
    for k in meta_data_na_dict:
        m_ori = meta_data_ori_dict[k]
        m_na = meta_data_na_dict[k]
        file_path = os.path.join(source_code_dir, m_na['real_proj_path'], m_na['completion_path'])
        if os.path.exists(file_path) is False:
            print(f"File does not exist: {file_path}")
        else:
            with open(file_path, 'w') as f:
                f.write(m_ori['file_content'])

def get_all_file_paths(root_dir):
    return [str(p) for p in Path(root_dir).rglob('*') if p.is_file()]         
            
def link_repo_source_code():
    files = get_all_file_paths('/root/workspace/code/CoderEval/docker_mount_out_data/python/repos')
    meta_data = utils.load_json_data('data/CoderEval/metadata.jsonl')
    
    def guess_real_paths(proj_path, c_path):
        ps = []
        for f in files:
            if proj_path in f and c_path in f and 'build' not in f:
                ps.append(f)
        return ps
    
    project_root_dict = {}
    
    for m in meta_data:
        proj_path = proj2path[m['project_path']]
        ps = guess_real_paths(proj_path, m['completion_path'])
        
        
        if len(ps) > 1:
            print(f'''repos: {m['project_path']}, completion_path: {m['completion_path']}, {m['namespace_real']}''')
            for p in ps:
                print(f'''\t{p}''')
            print('*'*10)
        if len(ps) == 0:
            print(f'''not found: repos: {m['project_path']}, completion_path: {m['completion_path']}, {m['namespace_real']}''')
            print('#'*10)
            
        project_root_dict[m['namespace']] = ps[0]
    return project_root_dict
        
if __name__ == '__main__':
    # create_dummy_source_code()
    # pass
    # create_meta_data_4_coderEval()
    # link_repo_source_code()
    copy_content_to_source_code_dir()
    
    
    
    
