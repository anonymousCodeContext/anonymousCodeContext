import os
import pprint
from typing import List, Set, Optional

from codebase_indexer import CodebaseIndexer

# Reuse accurate function discovery from gen_static
from gen_static import get_all_functions_in_file, get_target_function_id


def _print_directory_tree(start_path: str, max_depth: int = 2) -> str:
    """
    Render a directory tree view starting at start_path, limited by max_depth.
    Excludes common noisy folders.
    """
    exclude = {"venv", "env", ".git", "__pycache__", "tests", "myenv", "dist", "build", ".vscode", ".idea"}

    def _walk(path: str, prefix: str, depth: int) -> str:
        if depth > max_depth:
            return ""
        try:
            entries = sorted(os.listdir(path), key=lambda x: (not os.path.isdir(os.path.join(path, x)), x))
        except Exception:
            return ""
        out = ''
        for i, entry in enumerate(entries):
            if entry in exclude or entry.startswith('.'):
                continue
            is_last = i == len(entries) - 1
            entry_path = os.path.join(path, entry)
            connector = '└── ' if is_last else '├── '
            out += f"{prefix}{connector}{entry}\n"
            if os.path.isdir(entry_path):
                extension = '    ' if is_last else '│   '
                out += _walk(entry_path, prefix + extension, depth + 1)
        return out

    return _walk(start_path, '', 1)


def _extract_other_functions_code(project_root: str, target_file_path: str, target_function_id: str) -> List[str]:
    """
    From the target file, collect source code for all functions/methods except the target one.
    Returns a list of labeled code blocks.
    """
    # Build/Reuse indexer
    global _GLOBAL_INDEXER
    try:
        _GLOBAL_INDEXER
    except NameError:
        _GLOBAL_INDEXER = CodebaseIndexer()
    indexer = _GLOBAL_INDEXER
    indexer.ensure_indexed(project_root)

    rel_file = os.path.relpath(target_file_path, project_root)
    rel_file_posix = rel_file.replace(os.sep, '/')

    all_funcs = get_all_functions_in_file(target_file_path, project_root)

    target_simple_name = target_function_id.split('::')[-1]
    blocks: List[str] = []
    seen: Set[str] = set()

    for fid in all_funcs:
        if fid == target_function_id:
            continue
        # Skip any function with the same simple name as target to avoid leakage
        if fid.split('::')[-1] == target_simple_name:
            continue

        parts = fid.split('::')
        # parts like [path, func] or [path, Class, func]
        if len(parts) == 2:
            _, func_name = parts
            class_name: Optional[str] = None
        else:
            _, class_name, func_name = parts[-3:]

        code = indexer.get_function_code(target_file_path, func_name, class_name)
        if code and code not in seen:
            blocks.append(f"# Source for: {rel_file_posix}::" + (f"{class_name}::" if class_name else "") + f"{func_name}\n{code}\n")
            seen.add(code)

    return blocks


def generate_navigation_context(project_root: str, target_file_path: str, target_function_id: str, include_tree: bool = True) -> str:
    """
    Generate context using a simple Navigation heuristic:
    - Optional: a shallow project directory tree
    - All other functions/methods in the same file (excluding the target)
    """
    print(f"🧭 Generating Navigation context for: {target_file_path}")
    print(f"🎯 Excluding target function: {target_function_id}")

    context_chunks: List[str] = []

    if include_tree:
        context_chunks.append("# Project tree (partial)\n" + _print_directory_tree(project_root, max_depth=2))

    # Add other functions from the same file
    other_funcs = _extract_other_functions_code(project_root, target_file_path, target_function_id)
    # Prepend a brief summary header as comments
    try:
        rel = os.path.relpath(target_file_path, project_root).replace(os.sep, '/')
        summary = [
            f"# Summary: Navigation context",
            f"# Target file: {rel}",
            f"# Included snippets: {len(other_funcs)}",
        ]
        context_chunks.insert(0, "\n".join(summary) + "\n")
    except Exception:
        pass
    if other_funcs:
        context_chunks.extend(other_funcs)
    else:
        print("ℹ️ No other functions found in the target file or all filtered out.")

    return "\n".join(context_chunks)


def make_prompt(template_path='/root/workspace/code/anonymousCodeContext/prompt/template/sa_template_refine.txt', 
                element_path='/root/workspace/code/DevEval/prompt_elements_source_code2.jsonl', 
                meta_data_path='/root/workspace/code/anonymousCodeContext/data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                depandancy_dir='/root/workspace/code/anonymousCodeContext/data/DevEval/navigation',
                output_path='/root/workspace/code/anonymousCodeContext/data/DevEval/prompt_navigation.jsonl'):
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


def make_prompt_first_only(template_path='/root/workspace/code/anonymousCodeContext/prompt/template/sa_template_refine.txt', 
                           element_path='/root/workspace/code/DevEval/prompt_elements_source_code2.jsonl', 
                           meta_data_path='/root/workspace/code/anonymousCodeContext/data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                           depandancy_dir='/root/workspace/code/anonymousCodeContext/data/DevEval/navigation',
                           output_path='/root/workspace/code/anonymousCodeContext/data/DevEval/prompt_navigation_first.jsonl'):
    import utils
    meta_data = utils.load_json_data(meta_data_path)
    if not meta_data:
        raise ValueError('meta_data is empty')
    elem_dict = utils.load_json_data_as_dict(element_path)
    d = meta_data[0]
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
    with open(output_path, 'w') as f:
        f.write(json.dumps(elem) + '\n')


def save_navigation_ctxs(source_code_dir: str = '/root/workspace/code/DevEval/Source_Code2',
                         meta_data_path: str = '/root/workspace/code/anonymousCodeContext/data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                         output_dir: str = '/root/workspace/code/anonymousCodeContext/data/DevEval/navigation'):
    import utils
    meta_data = utils.load_json_data(meta_data_path)
    c = 0
    no_ctx = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for d in meta_data:
        save_path = os.path.join(output_dir, d['namespace'] + '.txt')
        if os.path.exists(save_path):
            continue

        try:
            project_root = os.path.join(source_code_dir, d['project_path'])
            target_file = os.path.join(source_code_dir, d['completion_path'])
            target_function_id = get_target_function_id(d['completion_path'], d['project_path'], d['namespace'])

            ctx = generate_navigation_context(project_root, target_file, target_function_id)

            if not ctx:
                print("No context was generated.", d['namespace'])
                no_ctx.append(d['namespace'])
            else:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(ctx)
            c += 1
        except Exception as e:
            print(f"❌ Error analyzing {d['namespace']}: {e}")

    print('total: ', c)
    print('no_ctx: ', len(no_ctx))
    print(no_ctx)


def save_navigation_ctxs4coder_eval(source_code_dir: str = '/root/workspace/code/CoderEval/docker_mount_out_data/python/repos',
                                    meta_data_path: str = '/root/workspace/code/anonymousCodeContext/data/CoderEval/metadata.jsonl',
                                    output_dir: str = '/root/workspace/code/anonymousCodeContext/data/CoderEval/navigation'):
    from coder_eval_fit import get_project_root_dict
    import utils

    project_root_dict = get_project_root_dict()
    meta_data = utils.load_json_data(meta_data_path)
    c = 0
    no_ctx = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for d in meta_data:
        save_path = os.path.join(output_dir, d['namespace'] + '.txt')
        if os.path.exists(save_path):
            continue

        try:
            r_proj_root = project_root_dict[d['namespace']]
            project_root = os.path.join(source_code_dir, r_proj_root)
            target_file = os.path.join(project_root, d['completion_path'])

            # Construct target_function_id similar to gen_static.save_sa_ctxs4coder_eval
            last = d['namespace_real'][len(d['completion_path']) - 2:]
            target_function_id = d['completion_path'] + '::' + last

            ctx = generate_navigation_context(project_root, target_file, target_function_id)
            if not ctx:
                print("No context was generated.", d['namespace'])
                no_ctx.append(d['namespace'])
            else:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(ctx)
            c += 1
        except Exception as e:
            print(f"❌ Error analyzing {d['namespace']}: {e}")

    print('total: ', c)
    print('no_ctx: ', len(no_ctx))
    print(no_ctx)


if __name__ == '__main__':
    # DevEval
    # save_navigation_ctxs()
    # CoderEval
    # save_navigation_ctxs4coder_eval()
    pass


