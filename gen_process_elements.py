import json
import os
import ast

Deveval_PATH = '/root/workspace/code/DevEval'
DevEvalSource_Benchmark2 = './data/DevEval/data_fixed2.jsonl'
DevEvalSource_PATH = '/root/workspace/code/DevEval/Source_Code2'
DevEvalSource_Benchmark = '/root/workspace/code/DevEval/data_fixed.jsonl'

def get_intersection():
    dev_set = set()
    evo_set = set()
    with open(DevEvalSource_Benchmark, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            dev_set.add(namespace)
    with open('prompt/prompt_elements.jsonl', 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            evo_set.add(namespace)

    seti = dev_set.intersection(evo_set) 
    print(len(seti))
    for k in evo_set:
        print(k)        


def check_class_and_method(code, target_class, target_method):    
    tree = ast.parse(code)
    class_exists = False
    method_exists = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == target_class:
            class_exists = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target_method:
                    method_exists = True
                    break
    
    return class_exists, method_exists


def read_code(source_code_root='/root/workspace/code/DevEval/Source_Code2', json_obj=None):
    path = os.path.join(source_code_root, json_obj['completion_path'])
    with open(path, 'r') as f:
        code = f.read()
    code_lines = code.split('\n')
    code_lines = code_lines[json_obj['signature_position'][0] - 1:json_obj['body_position'][1]]
    space_num = len(code_lines[0]) - len(code_lines[0].lstrip(' '))
    if space_num > 0:
        tmp = []
        for line in code_lines:
            if len(line) > space_num:
                tmp.append(line[space_num:])
            else:
                tmp.append('')
        code_lines = tmp
    ground_true = '\n'.join(code_lines)
    return ground_true, ' ' * space_num
    
def na_gen_prompt_elements(source_code_root=DevEvalSource_PATH,
                           meta_data_path=DevEvalSource_Benchmark2,
                           output_path='./data/DevEval/prompt_elements_source_code2.jsonl'):
    indent = '    '
    result = []
    with open(meta_data_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            
            with open(os.path.join(source_code_root, js['completion_path'])) as f:
                source_code = f.readlines()
            func_signature = js['signature']
            try:
                function_name = func_signature[func_signature.index('def ')+4 : func_signature.index('(')]
            except Exception as e:
                print(e)
            requirement = js['requirement']
            functionality = requirement['Functionality'].replace('\n', '\n'+indent)
            arguments = requirement['Arguments'].replace('\n', '\n'+indent)
            input_code = f'''{func_signature}
    """
    {functionality}
    {arguments}
    """
    '''
            sign_start = js['signature_position'][0] - 1
            body_end = js['body_position'][1] - 1
            contexts_above = ''.join(source_code[:sign_start])
            context_below = ''.join(source_code[body_end + 1:])
            class_name = None
            if js['type'] != 'function':
                class_name = js['namespace'].split('.')[-2]
                func_name = js['namespace'].split('.')[-1]
            ground_truth, space = read_code(source_code_root, js)
            e = {
                "namespace": js["namespace"],
                "type": js['type'],
                "class_name" : class_name, 
                "function_name": function_name,
                "contexts_above": contexts_above,
                "contexts_below": context_below,
                "input_code": input_code,
                "indent_space": space,
                "ground_truth": ground_truth,
                "signature": js['signature'],
                "completion_path": js['completion_path']
            }
            result.append(e)

        with open(output_path, 'w') as fout:
            for item in result:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        
def raw_gen_prompt_elements(meta_data_path, source_code_path, output_path):
    indent = '    '
    result = []
    with open(meta_data_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            
            with open(os.path.join(source_code_path, js['completion_path'])) as f:
                source_code = f.readlines()
            func_signature = js['signature']
            try:
                function_name = func_signature[func_signature.index('def ')+4 : func_signature.index('(')]
            except Exception as e:
                print(e)
            requirement = js['requirement']
            functionality = requirement['Functionality'].replace('\n', '\n'+indent)
            arguments = requirement['Arguments'].replace('\n', '\n'+indent)
            input_code = f'''{func_signature}
    """
    {functionality}
    {arguments}
    """
            '''
            sign_start = js['signature_position'][0] - 1
            body_end = js['body_position'][1] - 1
            contexts_above = ''.join(source_code[:sign_start])
            context_below = ''.join(source_code[body_end + 1:])
            class_name = None
            if js['type'] != 'function':
                class_name = js['namespace'].split('.')[-2]
                func_name = js['namespace'].split('.')[-1]
            sing_line = source_code[sign_start]
            space = sing_line[:len(sing_line) - len(sing_line.lstrip(' '))]
            e = {
                "namespace": js["namespace"],
                "type": js['type'],
                "class_name" : class_name, 
                "function_name": function_name,
                "contexts_above": contexts_above,
                "contexts_below": context_below,
                "input_code": input_code,
                "indent_space": space,
            }
            result.append(e)
    
        with open(output_path, 'w') as fout:
            for item in result:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return True                

# def pylint_check(path):
#     from pylint.lint import Run
#     from pylint.reporters import JSONReporter

    # from pylint import run_pylin
    
# import py_compile

def cp_min_prompt_elements(n=3):
    src_file = './prompt/navigation_deveval_prompt_elements.jsonl'
    dst_file = f'./prompt/navigation_deveval_prompt_elements_min_{n}.jsonl'
    with open(src_file, 'r') as fin, open(dst_file, 'w') as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)

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

    return result, error
     

def test_check_code_syntax():
    with open('./syntax_check.py', 'r') as f:
        code = ''
        for l in f.readlines():
            code += l
        
    flag, e = check_code_syntax(code)
    if not flag:
        print("Syntax error:", e)
    else:
        print("No syntax errors")


@DeprecationWarning
def refine_position_in_deveval_data():
    import tokenize
    import os
    indent = '    '
    fc = 0
    with open(DevEvalSource_Benchmark2, 'r') as f:
        for ki, line in enumerate(f):
            js = json.loads(line)
            
            target_method_name = js['namespace'].split('.')[-1]
            
            
            bs, be = js['signature_position'][0], js['signature_position'][1]
            min_d = 10000
            m_token = None
            with open(os.path.join(DevEvalSource_PATH, js['completion_path'])) as f:
                source_code = f.readlines()
                sc = ''.join(source_code)

            with tokenize.open(os.path.join(DevEvalSource_PATH, js['completion_path'])) as f:
                tokens = tokenize.generate_tokens(f.readline)
                for token in tokens:
                    if token.type == tokenize.NAME:
                        if token.string == 'def':
                            method_tn = next(tokens)
                            start = method_tn.start
                            d = abs(start[0]-bs)
                            if d < min_d:
                                min_d = d
                                m_token = method_tn
                
                # if m_token is None:
                #     print(ki, js['namespace'])
                #     exit()
                
                r_start = m_token.start[0] - 1
                r_end = r_start + (be - bs) # 
                
                if not source_code[r_end].rstrip('\n').endswith(':'):
                    for ei in range(r_start, r_end+1):
                        tmp = source_code[ei].lstrip(' ')
                        if tmp.startswith('#') or \
                            tmp.startswith("""'''""") or \
                            tmp.startswith('''"""''') or \
                            tmp.startswith('\n') or \
                            tmp.startswith('@'):
                            r_end = ei - 1
                            break
                    
                if not source_code[r_end].rstrip('\n').endswith(':'):
                    pass

                
                
                signature = (''.join(source_code[r_start: r_end+1])).rstrip('\n').rstrip(' ').lstrip(' ')
                if not signature.endswith(':'):
                    signature_end = signature.find('#')
                    signature = signature[: signature_end]
                    signature_end = signature.rfind(':')
                    signature = signature[:signature_end + 1]
                
                
                body_start, body_end =  js['body_position'][0] - 1, js['body_position'][1] - 1
                body_start += min_d
                body_end += min_d
                
                indent_tmp = len(source_code[body_start]) - len(source_code[body_start].lstrip(' '))
                indent_str = source_code[body_start][:indent_tmp]
                sc_tmp = source_code[:body_start] + ['\n', indent_str+'pass', '\n'] + source_code[body_end+1:]
                
                code_tmp = ''.join(sc_tmp)
                flag, e = check_code_syntax(code_tmp)
                if not flag:
                    if '@wraps' in source_code[body_start - 1]:
                        continue
                    fc += 1
                    print(ki, js)
                    print('------------------ body position:', body_start, body_end)
                    print('------------------ file:', os.path.join(DevEvalSource_PATH, js['completion_path']))
                    print("Syntax error:", e)
                    print(code_tmp)
                    exit()
                
                
                continue
                
                if min_d == 0:

                    
                    ''' =========Regarding signature======: based on min_d=0: clearly does not include the comment part'''
                    print('=============new=============')
                    print('path:', os.path.join(DevEvalSource_PATH, js['completion_path']))
                    print('token:', m_token)
                    
                    data = js
                    sos, eos = data['body_position'][0]-1, data['body_position'][1]
                    file_lines = source_code
                    print('-----------------:signature:')
                    print(''.join(file_lines[r_start: r_end]))
                    print('-----------------source code:')
                    print(''.join(file_lines[r_start: eos+2]))
                    
                    fc += 1
                    if fc > 10:
                        exit()    
                # break
                
    print(fc)
    
def check_position_in_deveval_data(meta_path=DevEvalSource_Benchmark2, 
                                   source_code_path=DevEvalSource_PATH):
    import os
    fc = 0
    with open(meta_path, 'r') as f:
        for ki, line in enumerate(f):
            js = json.loads(line)
            target_method_name = js['namespace'].split('.')[-1]
            
            signature = js['signature']
            if target_method_name not in signature or \
                not (signature.startswith('def') or signature.startswith('async') or signature.startswith('@')):
                print('Signature error:')
                print(ki, js['namespace'])
                print('signature:', signature)
                fc += 1
            
            
            body_start, body_end =  js['body_position'][0] - 1, js['body_position'][1] - 1
            with open(os.path.join(source_code_path, js['completion_path'])) as f:
                source_code = f.readlines()
            indent_tmp = len(source_code[body_start]) - len(source_code[body_start].lstrip(' '))
            indent_str = source_code[body_start][:indent_tmp]
            sc_tmp = source_code[:body_start] + ['\n', indent_str+'pass', '\n'] + source_code[body_end+1:]
            
            code_tmp = ''.join(sc_tmp)
            flag, e = check_code_syntax(code_tmp)
            if not flag:
                if '@wraps' in source_code[body_start - 1]:
                    continue
                print('Body position error:')    
                fc += 1
                print(ki, js)
                print('------------------ body position:', body_start, body_end)
                print('------------------ file:', os.path.join(source_code_path, js['completion_path']))
                print("Syntax error:", e)
    print(fc)    

def fix_position_in_deveval_data():
    import tokenize
    import os
    indent = '    '
    fc = 0
    result = []
    with open(DevEvalSource_Benchmark, 'r') as f:
        for ki, line in enumerate(f):
            js = json.loads(line)
            
            target_method_name = js['namespace'].split('.')[-1]
            
            bs, be = js['signature_position'][0], js['signature_position'][1]
            min_d = 10000
            m_token = None
            with open(os.path.join(DevEvalSource_PATH, js['completion_path'])) as f:
                source_code = f.readlines()
                sc = ''.join(source_code)

            with tokenize.open(os.path.join(DevEvalSource_PATH, js['completion_path'])) as f:
                tokens = tokenize.generate_tokens(f.readline)
                for token in tokens:
                    
                    if token.type == tokenize.NAME:
                        if token.string == 'def':
                            method_tn = next(tokens)
                            start = method_tn.start
                            d = abs(start[0]-bs)
                            if d < min_d:
                                min_d = d
                                m_token = method_tn
                
                
                r_start = m_token.start[0] - 1
                r_end = r_start + (be - bs) # 
                
                if not source_code[r_end].rstrip('\n').endswith(':'):
                    for ei in range(r_start, r_end+1):
                        tmp = source_code[ei].lstrip(' ')
                        if tmp.startswith('#') or \
                            tmp.startswith("""'''""") or \
                            tmp.startswith('''"""''') or \
                            tmp.startswith('\n') or \
                            tmp.startswith('@'):
                            r_end = ei - 1
                            break
                    

                signature = (''.join(source_code[r_start: r_end+1])).rstrip('\n').rstrip(' ').lstrip(' ')
                if not signature.endswith(':'):
                    signature_end = signature.find('#')
                    signature = signature[: signature_end]
                    signature_end = signature.rfind(':')
                    signature = signature[:signature_end + 1]
                
                
                body_start, body_end =  js['body_position'][0] - 1, js['body_position'][1] - 1
                body_start += min_d
                body_end += min_d
                
                js['signature_position'] = [r_start + 1, r_end + 1]
                js['body_position'] = [body_start + 1, body_end + 1]
                js['signature'] = signature
                result.append(js)
    
        with open(DevEvalSource_Benchmark2, 'w') as fout:
            for item in result:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                
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


def get_above_below_by_code(code: str, target_method_name: str):
    tree = ast.parse(code)
    code_lines = code.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
                temp_method_name = node.name
                if target_method_name != temp_method_name and "_"+target_method_name!=temp_method_name:
                    continue
                start_line = node.lineno
                end_line = node.end_lineno
                above = '\n'.join(code_lines[:start_line - 1])
                below = '\n '.join(code_lines[end_line + 1:]) if end_line is not None else ''
                node.body = [ast.Pass()]
                return above, below, ast.unparse(node)            
    return None, None, ''


def get_function_signature(code_str: str, func_name: str) -> str:
    """Removes the function body using AST parsing"""
    tree = ast.parse(code_str)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.Pass()]
            return ast.unparse(node)
    raise ValueError(f"Function '{func_name}' not found")




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
 
 
def find_java_method_lines_by_name_and_lineno(code, method_name, lineno):
    """
    Analyzes Java source code using tree_sitter to find the start line, end line, and body start line
    of the method with the specified name that is closest to the given lineno.
    :param code: Java source code string
    :param method_name: Method name string
    :param lineno: Target line number (int, 1-based)
    :return: A tuple (start_line, end_line, body_start_line), all 1-based line numbers, or None if not found
    """
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava

    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    candidates = []
    
    code_lines = code.split('\n')

    def traverse(node):
        if node.type == 'method_declaration':
            id_node = None
            for child in node.children:
                if child.type == 'identifier':
                    id_node = child
                    break
            if child.type == 'identifier':
                if method_name in code_lines[node.start_point[0]]:
                    print(code_lines[node.start_point[0]])
                name = code[child.start_byte:child.end_byte]
                if name == method_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    body_start_line = None
                    for c in node.children:
                        if c.type == 'block':
                            body_start_line = c.start_point[0] + 1
                            break
                    candidates.append((start_line, end_line, body_start_line))
        for child in node.children:
            traverse(child)

    traverse(root_node)

    if not candidates:
        return None, None, None

    def dist(x):
        s, e, _ = x
        if s <= lineno <= e:
            return 0
        return min(abs(s - lineno), abs(e - lineno))

    candidates.sort(key=dist)
    print(candidates[0])
    return candidates[0]

            
def gen_prompt_elements4coder_eval_java():
    code_json_path = '/root/workspace/code/CoderEval/docker_mount_out_data/java/CoderEval4Java.json'    
    def find_method_by_name(node, target_name):
        start_line, end_line, body_start_line = None, None, None
        if node.type == 'method_declaration':
            for child in node.children:
                if child.type == 'identifier':
                    method_name = code[child.start_byte:child.end_byte]
                    print(method_name)
                    if method_name == target_name:
                        print(method_name, 'in')
                        start_line = node.start_point[0] + 1
                        end_line = node.end_point[0] + 1
                        body_start_line = None
                        for c in node.children:
                            if c.type == 'block':
                                body_start_line = c.start_point[0] + 1
                                break
                        return  start_line, end_line, body_start_line
        for child in node.children:
            start_line, end_line, body_start_line = find_method_by_name(child, target_name)
            if start_line is not None:
                break
        return start_line, end_line, body_start_line
    
    f = open(code_json_path, 'r', encoding="utf-8")
    content = f.read()
    f.close()
    collection = []
    collection_dictt = {}
    content_json = json.loads(content)
    
    
    for i, l in enumerate(content_json['RECORDS']):
        if l['_id'] != '636766f61a6d9265ec017701':
            continue
        code = l['file_content']
        function_name = l['name']
        start_line, end_line, body_start_line = find_java_method_lines_by_name_and_lineno(code, function_name, int(l['lineno']))
        if start_line is None or end_line is None or body_start_line is None:
            print(f"Method {l['name']} start_line or end_line or body_start_line is None")
            print(function_name)
            print(f'Start line: {start_line}')
            print(f'End line: {end_line}')
            print(f'Body start line: {body_start_line}')
            continue
        code_lines = code.split('\n')
        above = '\n'.join(code_lines[:start_line - 1])
        docstring = l['docstring']
        if above.endswith(docstring):
            above = above[:-len(docstring)]
        below = '\n '.join(code_lines[end_line:]) if body_start_line is not None else ''
        signature = '\n '.join(code_lines[start_line - 1: body_start_line]).rstrip().rstrip('{')
        assert function_name in signature

    
    

if __name__ == '__main__':
    # main()
    # refine_position_in_deveval_data()
    # test_check_file_syntax()
    # test_check_code_syntax()
    # fix_position_in_deveval_data()
    # check_position_in_deveval_data()
    # cp_min_prompt_elements()
    # cp_min_prompt_elements(5)
    # cp_min_prompt_elements(1)
    # gen_prompt_elements4coder_eval_python()
    # gen_prompt_elements4coder_eval_java()
    
    # check_position_in_deveval_data('data/CoderEval/metadat_coderEval.jsonl',
    #                                source_code_path='data/CoderEval/dummy_source_code')
    
    na_gen_prompt_elements(source_code_root='/root/workspace/code/DevEval/Source_Code2',
                           meta_data_path='data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                           output_path='data/DevEval/prompt_elements_source_code2_sample_per_proj.jsonl')