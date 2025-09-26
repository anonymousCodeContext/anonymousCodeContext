from argparse import Namespace
import ast
import os
import pprint
from typing import List, Set, Dict, Optional
import json
from ast_analyzer.analyzer import AstAnalyzer
from codebase_indexer import CodebaseIndexer

def read_line_from_file(file_path: str, line_number: int) -> Optional[str]:
    """Reads a specific line from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_number - 1:
                    return line.strip()
        return None
    except FileNotFoundError:
        return None

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, relative_path):
        self.relative_path = relative_path
        self.functions = []
        self._current_class = None

    def visit_ClassDef(self, node):
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = None

    def visit_FunctionDef(self, node):
        if self._current_class:
            self.functions.append(f"{self.relative_path}::{self._current_class}::{node.name}")
        else:
            self.functions.append(f"{self.relative_path}::{node.name}")
        self.generic_visit(node)

def get_all_functions_in_file(file_path: str, project_root: str) -> List[str]:
    """
    Parses a Python file and returns a list of all function and method identifiers
    using a NodeVisitor for accuracy.

    Args:
        file_path: The absolute path to the Python file.
        project_root: The absolute path to the project's root directory.

    Returns:
        A list of function identifiers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content, filename=file_path)
    relative_path = os.path.relpath(file_path, project_root).replace(os.sep, '/')
    
    visitor = FunctionVisitor(relative_path)
    visitor.visit(tree)
    return visitor.functions

def generate_context_for_file(project_root: str, target_file_path: str, target_function_id: str) -> str:
    """
    Generates a context string containing the source code of all dependencies of functions
    in a file, excluding the target function itself.

    Args:
        project_root: The absolute path to the project's root directory.
        target_file_path: The absolute path to the target Python file.
        target_function_id: The identifier of the function to be excluded from analysis.

    Returns:
        A string containing the concatenated source code of all found dependencies.
    """
    print(f"🔬 Analyzing file: {target_file_path}")
    print(f"🎯 Excluding target function: {target_function_id}")

    # Extract the simple name of the target function to filter dependencies
    target_function_name = target_function_id.split('::')[-1]

    # 1. Get all functions in the file
    all_funcs_in_file = get_all_functions_in_file(target_file_path, project_root)
    
    # 2. Exclude the target function
    functions_to_analyze = [f for f in all_funcs_in_file if f != target_function_id]
    
    print("\nFunctions to analyze:")
    pprint.pprint(functions_to_analyze)

    # 3. Analyze dependencies for the remaining functions
    analyzer = AstAnalyzer(project_root)
    all_dependencies: Set[str] = set()

    for func_id in functions_to_analyze:
        try:
            # The analyzer now supports both 'path::class::func' and 'path::func'
            analysis_result = analyzer.analyze_function(func_id)
            for dep in analysis_result.dependencies:
                all_dependencies.add(dep) # Add the whole Dependency object
        except Exception as e:
            print(f"❌ Error analyzing {func_id}: {e}")
            
    print("\nFound dependencies:")
    pprint.pprint(all_dependencies)

    # Filter out dependencies with the same name as the target function to avoid data leakage
    original_dep_count = len(all_dependencies)
    all_dependencies = {
        dep for dep in all_dependencies
        if dep.name.split('.')[-1] != target_function_name
    }

    # 4. Index the codebase and retrieve dependency source code
    # Reuse a global indexer if available to avoid repeated indexing cost
    global _GLOBAL_INDEXER
    try:
        _GLOBAL_INDEXER
    except NameError:
        _GLOBAL_INDEXER = CodebaseIndexer()
    indexer = _GLOBAL_INDEXER
    indexer.ensure_indexed(project_root)
    
    context_code = []
    processed_code_blocks = set()  # To track added code blocks and prevent duplicates
    print("\n📚 Retrieving source code for dependencies...")

    for dep in sorted(list(all_dependencies), key=lambda d: d.name):
        dep_name = dep.name
        code = None  # Initialize code for this dependency

        if dep.dependency_type in ('property', 'cross_file_property'):
            parts = dep_name.split('.')
            prop_name = parts[-1]
            class_name_fq = ".".join(parts[:-1])
            class_name_simple = class_name_fq.split('.')[-1]
            module_path = ".".join(class_name_fq.split('.')[:-1])
            file_path_from_module = os.path.join(project_root, module_path.replace('.', os.sep) + '.py')

            if os.path.exists(file_path_from_module):
                code = indexer.get_function_code(file_path_from_module, prop_name, class_name_simple)

        elif dep.dependency_type == 'attribute_access':
            # This is likely an attribute. e.g., 'mrjob.runner.MRJobRunner.fs'
            parts = dep_name.split('.')
            attribute_name = parts[-1]
            class_name_fq = ".".join(parts[:-1])
            
            # Use the new analyzer method to find the definition
            definition_dep = analyzer.find_attribute_definition(class_name_fq, attribute_name)

            if definition_dep:
                # We found the definition, let's get the line of code
                code = read_line_from_file(definition_dep.file_path, definition_dep.line_no)
                # Heuristic: also try to pull the class implementation referenced on the RHS
                # e.g., self.fs = CompositeFilesystem(...)
                if code:
                    import re
                    # capture tokens that look like ClassName(
                    guessed_classes = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", code))
                    # filter obvious non-class function words
                    blacklist = {attribute_name, 'self', 'dict', 'list', 'set', 'tuple'}
                    guessed_classes = {g for g in guessed_classes if g not in blacklist and g[0].isupper()}
                    for guess in guessed_classes:
                        # attempt to find and add the class code anywhere in project
                        for indexed_path in list(indexer.trees.keys()):
                            class_code = indexer.get_class_code(indexed_path, guess)
                            if class_code and class_code not in processed_code_blocks:
                                print(f"✅ Found class code for guessed type: {guess} in {indexed_path}")
                                context_code.append(f"# Source for: {os.path.splitext(os.path.relpath(indexed_path, project_root))[0].replace(os.sep, '.')}\n{class_code}\n")
                                processed_code_blocks.add(class_code)
        
        elif dep.dependency_type == 'module_variable':
            parts = dep_name.split('.')
            variable_name = parts[-1]
            module_name = ".".join(parts[:-1])
            file_path_from_module = os.path.join(project_root, module_name.replace('.', os.sep) + '.py')
            code = indexer.get_module_variable_code(file_path_from_module, variable_name)

        else: # Default to function/class resolution
            parts = dep_name.split('.')
            func_name = parts[-1]
            class_name = None
            module_path = ""
            if len(parts) > 1 and parts[-2][0].isupper():
                class_name = parts[-2]
                module_path = ".".join(parts[:-2])
            else:
                module_path = ".".join(parts[:-1])
            
            file_path_from_module = os.path.join(project_root, module_path.replace('.', os.sep) + '.py')

            if os.path.exists(file_path_from_module):
                code = indexer.get_function_code(file_path_from_module, func_name, class_name)
                if not code and not class_name:
                    code = indexer.get_class_code(file_path_from_module, func_name)

        # --- Add code to context if found and not a duplicate ---
        if code:
            if code not in processed_code_blocks:
                print(f"✅ Found code for: {dep_name}")
                context_code.append(f"# Source for: {dep_name}\n{code}\n")
                processed_code_blocks.add(code)
            else:
                print(f"ℹ️  Skipping duplicate code for: {dep_name}")
        else:
             print(f"❌ Could not retrieve code for: {dep_name}")
            
    return "\n".join(context_code)

def get_target_function_id(completion_path, project_path, namespace):
    relative_path = completion_path[len(project_path):]
    tp = relative_path[:-3]
    post = namespace[len(tp):].replace('.', '::')
    return relative_path + '::' + post

def make_prompt(template_path='/root/workspace/code/EvoCodeBench/prompt/template/sa_template_refine.txt', 
                element_path='/root/workspace/code/DevEval/prompt_elements_source_code2.jsonl', 
                meta_data_path='/root/workspace/code/EvoCodeBench/data/DevEval/data_kl_fixed2_sample_pre_proj.jsonl',
                depandancy_dir='/root/workspace/code/EvoCodeBench/data/DevEval/similarity/sa',
                output_path='/root/workspace/code/EvoCodeBench/data/DevEval/prompt_sa.jsonl'):
    import utils
    meta_data = utils.load_json_data(meta_data_path)
    elem_dict = utils.load_json_data_as_dict(element_path)
    
    with open(output_path, 'w') as f:
        for d in meta_data:
            template = open(template_path, 'r').read()
            elem = elem_dict[d['namespace']]
            dep_path = os.path.join(depandancy_dir, d['namespace'] + '.txt')
            dep_str = '# No dependency found'
            if os.path.exists(dep_path):
                dep_str = open(dep_path, 'r').read()

            prompt = template.format(
                function_name=elem['function_name'],
                contexts_above=elem['contexts_above'],
                contexts_below=elem['contexts_below'],
                input_code=elem['input_code'],
                dependencies=dep_str
            )
            del elem['contexts_above']
            del elem['contexts_below']
            elem['prompt'] = prompt
            f.write(json.dumps(elem) + '\n')

def save_sa_ctxs(source_code_dir='/root/workspace/code/DevEval/Source_Code2', 
                meta_data_path='/root/workspace/code/EvoCodeBench/data/DevEval/data_kl_fixed2_sample_pre_proj.jsonl',
                output_dir='/root/workspace/code/EvoCodeBench/data/DevEval/sa'):
    import utils
    meta_data = utils.load_json_data(meta_data_path)
    c = 0
    no_ctx = []
    for d in meta_data:
        save_path = os.path.join(output_dir, d['namespace'] + '.txt')
        if os.path.exists(save_path):
            continue
        try:
            project_root = os.path.join(source_code_dir, d['project_path'])
            traget_file = os.path.join(source_code_dir, d['completion_path'])
            target_function_id = get_target_function_id(d['completion_path'], d['project_path'], d['namespace'])
            
            generated_context = generate_context_for_file(
                                project_root=project_root,
                                target_file_path=traget_file,
                                target_function_id=target_function_id)
            
            if not generated_context:
                print("No context was generated.", d['namespace'])
                no_ctx.append(d['namespace'])
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, d['namespace'] + '.txt')
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(generated_context)

            c += 1
            
        except Exception as e:
            print(f"❌ Error analyzing {d['namespace']}: {e}")
    
    print('total: ', c)
    print('no_ctx: ', len(no_ctx))
    print(no_ctx)
    
    
def test():
    completion_path = "System/mrjob/mrjob/hadoop.py"
    project_path = "project_path"
    namespace = "mrjob.hadoop.HadoopJobRunner._stream_history_log_dirs"
    print(get_target_function_id(completion_path, project_path, namespace))
    exit()
            
def save_sa_ctxs4coder_eval(source_code_dir='/root/workspace/code/CoderEval/docker_mount_out_data/python/repos', 
                meta_data_path='/root/workspace/code/EvoCodeBench/data/CoderEval/metadata.jsonl',
                output_dir='/root/workspace/code/EvoCodeBench/data/CoderEval/sa'):
    from coder_eval_fit import get_project_root_dict
    project_root_dict = get_project_root_dict()
    
    # for p in project_root_dict:
    #     print(project_root_dict[p])
        
    # exit()
    
    import utils
    meta_data = utils.load_json_data(meta_data_path)
    c = 0
    no_ctx = []
    for d in meta_data:
        save_path = os.path.join(output_dir, d['namespace'] + '.txt')
        if os.path.exists(save_path):
            continue
        try:
            r_proj_root = project_root_dict[d['namespace']]
            project_root = os.path.join(source_code_dir, r_proj_root)
            traget_file = os.path.join(project_root, d['completion_path'])
            
            last = d['namespace_real'][len(d['completion_path']) - 2:]
            target_function_id = d['completion_path'] + '::' + last
            
            generated_context = generate_context_for_file(
                                project_root=project_root,
                                target_file_path=traget_file,
                                target_function_id=target_function_id)
            
            if not generated_context:
                print("No context was generated.", d['namespace'])
                no_ctx.append(d['namespace'])
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, d['namespace'] + '.txt')
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(generated_context)

            c += 1
            
        except Exception as e:
            print(f"❌ Error analyzing {d['namespace']}: {e}")
    
    print('total: ', c)
    print('no_ctx: ', len(no_ctx))
    print(no_ctx)

if __name__ == '__main__':
    # make_prompt()
    # save_sa_ctxs()
    save_sa_ctxs4coder_eval()
    exit()
    # test()
    # Test case from get_static.py for a real project
    # PROJECT_ROOT = os.path.abspath('/Users/liyichen/Downloads/Source_Code/System/mrjob/')
    PROJECT_ROOT = os.path.abspath('/root/workspace/code/DevEval/Source_Code2/System/mrjob/')
    TARGET_FILE = os.path.join(PROJECT_ROOT, 'mrjob/hadoop.py')
    TARGET_FUNCTION_ID = 'mrjob/hadoop.py::HadoopJobRunner::_stream_history_log_dirs'

    print("="*50)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"TARGET_FILE: {TARGET_FILE}")
    print(f"TARGET_FUNCTION_ID: {TARGET_FUNCTION_ID}")
    print("="*50)

    print("\n--- Running full context generation for real project ---")
    generated_context = generate_context_for_file(
        project_root=PROJECT_ROOT,
        target_file_path=TARGET_FILE,
        target_function_id=TARGET_FUNCTION_ID
    )
    
    print("\n--- Generated Context ---")
    if generated_context:
        print(generated_context)
    else:
        print("No context was generated.")
    print("--- End of Context ---")

    # Save the result to a file
    output_filename = 'analysis_result.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(generated_context)
    print(f"\n✅ Analysis result saved to {output_filename}")
