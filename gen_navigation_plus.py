import os
from typing import List, Set, Optional

from navigation_plus_python.tools import NavigationPlus
from codebase_indexer import CodebaseIndexer
from gen_static import get_all_functions_in_file, get_target_function_id


def _bm25_top_files(navigator: NavigationPlus, query: str, top_k: int = 5) -> List[str]:
    """
    Use NavigationPlus' BM25 to retrieve top related files for a simple query.
    Returns absolute paths.
    """
    results = navigator.semantic_search(query, top_k)
    files: List[str] = []
    for r in results:
        try:
            path = r.split(' (score:')[0].strip()
            files.append(path)
        except Exception:
            continue
    return files


def _collect_supporting_code(project_root: str,
                             target_file_path: str,
                             target_function_id: str,
                             top_related_files: List[str]) -> List[str]:
    """
    Gather supporting code blocks from target file (excluding target) and top related files.
    """
    global _GLOBAL_INDEXER
    try:
        _GLOBAL_INDEXER
    except NameError:
        _GLOBAL_INDEXER = CodebaseIndexer()
    indexer = _GLOBAL_INDEXER
    indexer.ensure_indexed(project_root)

    rel_target_file = os.path.relpath(target_file_path, project_root).replace(os.sep, '/')
    target_simple_name = target_function_id.split('::')[-1]

    blocks: List[str] = []
    seen: Set[str] = set()

    # 1) All other functions in the same file
    for fid in get_all_functions_in_file(target_file_path, project_root):
        if fid == target_function_id:
            continue
        if fid.split('::')[-1] == target_simple_name:
            continue
        parts = fid.split('::')
        if len(parts) == 2:
            _, func_name = parts
            class_name: Optional[str] = None
        else:
            _, class_name, func_name = parts[-3:]
        code = indexer.get_function_code(target_file_path, func_name, class_name)
        if code and code not in seen:
            label = f"# Source for: {rel_target_file}::" + (f"{class_name}::" if class_name else "") + f"{func_name}"
            blocks.append(f"{label}\n{code}\n")
            seen.add(code)

    # 2) From top related files, include top-level functions/classes (lightweight)
    for abs_path in top_related_files:
        if not abs_path.endswith('.py'):
            continue
        rel_path = os.path.relpath(abs_path, project_root).replace(os.sep, '/')
        try:
            for fid in get_all_functions_in_file(abs_path, project_root):
                # Avoid adding functions that share the same simple name as target
                if fid.split('::')[-1] == target_simple_name:
                    continue
                parts = fid.split('::')
                if len(parts) == 2:
                    _, func_name = parts
                    class_name = None
                else:
                    _, class_name, func_name = parts[-3:]
                code = indexer.get_function_code(abs_path, func_name, class_name)
                if code and code not in seen:
                    label = f"# Source for: {rel_path}::" + (f"{class_name}::" if class_name else "") + f"{func_name}"
                    blocks.append(f"{label}\n{code}\n")
                    seen.add(code)
        except Exception:
            continue

    return blocks


def generate_navigation_plus_context(project_root: str,
                                     target_file_path: str,
                                     target_function_id: str,
                                     bm25_top_k: int = 5) -> str:
    """
    Generate context with NavigationPlus:
    - Build an index over the project
    - Rank top related files by a simple query (target simple name)
    - Include non-target functions from target file and selected related files
    """
    print(f"🧭➕ Generating Navigation-Plus context for: {target_file_path}")
    print(f"🎯 Excluding target function: {target_function_id}")

    navigator = NavigationPlus(project_root)
    navigator.index()

    target_simple_name = target_function_id.split('::')[-1]
    related_files = _bm25_top_files(navigator, query=target_simple_name, top_k=bm25_top_k)

    blocks = _collect_supporting_code(project_root, target_file_path, target_function_id, related_files)

    return "\n".join(blocks)


def save_navigation_plus_ctxs(source_code_dir: str = '/root/workspace/code/DevEval/Source_Code2',
                              meta_data_path: str = '/root/workspace/code/anonymousCodeContext/data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                              output_dir: str = '/root/workspace/code/anonymousCodeContext/data/DevEval/navigation_plus'):
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

            ctx = generate_navigation_plus_context(project_root, target_file, target_function_id)
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


def save_navigation_plus_ctxs4coder_eval(source_code_dir: str = '/root/workspace/code/CoderEval/docker_mount_out_data/python/repos',
                                         meta_data_path: str = '/root/workspace/code/anonymousCodeContext/data/CoderEval/metadata.jsonl',
                                         output_dir: str = '/root/workspace/code/anonymousCodeContext/data/CoderEval/navigation_plus'):
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

            last = d['namespace_real'][len(d['completion_path']) - 2:]
            target_function_id = d['completion_path'] + '::' + last

            ctx = generate_navigation_plus_context(project_root, target_file, target_function_id)
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
    # save_navigation_plus_ctxs()
    # CoderEval
    # save_navigation_plus_ctxs4coder_eval()
    pass


