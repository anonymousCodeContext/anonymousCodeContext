from httpx import get
from langchain_core.tools import tool
import os
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from navigation.agentic_coder import AgentHelper

@tool(parse_docstring=True)
def read_file(task_id: str, file_path: str, start_line: int=None, end_line: int=None) -> str:
    """
    read the content of a file
    If start_line and end_line are provided, only the content between start_line and end_line will be returned.
    If start_line and end_line are not provided, the entire file will be returned.
    
    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        file_path: the path of the file to read
        start_line: the start line of the file to read
        end_line: the end line of the file to read

    Returns:
        the content of the file
    """
    state = AgentHelper.STATES[task_id]
    proj_path = state['project_path']
    a_file_path = os.path.join(proj_path, file_path)
    if os.path.exists(a_file_path) is False:
        return f"Error: File does not exist: {file_path}"
    elif os.path.isdir(a_file_path):
        return f"Error: {file_path} is a directory, not a file"
    
    with open(a_file_path, 'r') as file:
        lines = file.readlines()
        if start_line is not None and end_line is not None:
            return '\n'.join(lines[start_line-1:end_line])
        if start_line is not None and end_line is None:
            return '\n'.join(lines[start_line-1:])
        if start_line is None and end_line is not None:
            return '\n'.join(lines[:end_line])
        else:
            return '\n'.join(lines)
    
@tool(parse_docstring=True)
def list_dir(task_id: str, dir_path: str) -> list[str]:
    """
    list the files in a directory

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        dir_path: the path of the directory to list

    Returns:
        the list of files in the directory
    """
    state = AgentHelper.STATES[task_id]
    proj_path = state['project_path']
    a_dir_path = os.path.join(proj_path, dir_path)
    if os.path.exists(a_dir_path) is False:
        return f"Error: directory does not exist: {dir_path}"
    elif os.path.isfile(a_dir_path):
        return f"Error: {dir_path} is a file, not a directory"
    
    return os.listdir(a_dir_path)

@tool(parse_docstring=True)
def grep_search(task_id: str, file_path: str, pattern: str) -> str:
    """
    Performs content-based searches within a file using regular expressions or keyword matching.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        file_path: the path of the file to search
        pattern: the pattern to search for

    Returns:
        the matching content
    """
    state = AgentHelper.STATES[task_id]
    proj_path = state['project_path']
    a_file_path = os.path.join(proj_path, file_path)
    if os.path.exists(a_file_path) is False:
        return f"Error: File does not exist: {file_path}"
    elif os.path.isdir(a_file_path):
        return f"Error: {file_path} is a directory, not a file"

    try:
        with open(a_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    matches = []
    for idx, line in enumerate(lines, 1):
        if re.search(pattern, line):
            matches.append(f"Line {idx}: {line.strip()}")

    if matches:
        return "\n".join(matches)
    else:
        return "No matching content found."

@tool(parse_docstring=True)
def glob(task_id: str, dir_path: str, pattern: str) -> list[str]:
    """
    Searches for files based on naming patterns (e.g., globbing like src/**/*.py).

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        dir_path: the directory path to search in
        pattern: the filename pattern to match (supports wildcards, e.g., *.py, **/*.txt)

    Returns:
        a list of matched file paths
    """
    import glob as glob_module
    import os
    state = AgentHelper.STATES[task_id]
    proj_path = state['project_path']
    a_dir_path = os.path.join(proj_path, dir_path)
    if os.path.exists(a_dir_path) is False:
        return f"Error: directory does not exist: {dir_path}"
    elif os.path.isfile(a_dir_path):
        return f"Error: {dir_path} is a file, not a directory"
    
    search_pattern = os.path.join(a_dir_path, pattern)
    # recursive=True enables support for ** wildcard
    return glob_module.glob(search_pattern, recursive=True)

# from langchain_tavily import TavilySearch
# web_search = TavilySearch(tavily_api_key='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')

def print_directory_tree(startpath, prefix=''):
    entries = sorted(os.listdir(startpath), key=lambda x: (not os.path.isdir(os.path.join(startpath, x)), x))
    trees = ''
    for i, entry in enumerate(entries):
        if entry == 'myenv' or entry == 'tests' or entry == '__pycache__':
            continue
        is_last = i == len(entries) - 1
        entry_path = os.path.join(startpath, entry)
        if os.path.isdir(entry_path) and entry.startswith('.'):
            continue

        connector = '└── ' if is_last else '├── '
        # print(f"{prefix}{connector}{entry}")
        trees += f"{prefix}{connector}{entry}\r\n"
        if os.path.isdir(entry_path):
            extension = '    ' if is_last else '│   '
            trees += print_directory_tree(entry_path, prefix + extension)
    return trees

@tool(parse_docstring=True)
def read_dir_tree(task_id: str, path: str) -> str:
    """
    read the directory tree.
    This function will read the directory structure of the specified directory without reading the actual contents of the files.
    The output will be a string that represents the directory tree as follows:
    ├── src
    │   ├── __init__.py
    │   ├── a.py
    │   ├── utils
    │   │   ├── b.py
    │   │   ├── c.py
    │   │   └── d.py
    └── LICENSE

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        path: the path of the directory to read

    Returns:
        the directory tree
    """
    state = AgentHelper.STATES[task_id]
    proj_path = state['project_path']
    a_dir_path = os.path.join(proj_path, path)
    if os.path.exists(a_dir_path) is False:
        return f"Error: directory does not exist: {path}"
    elif os.path.isfile(a_dir_path):
        return f"Error: {path} is a file, not a directory"
    
    return print_directory_tree(a_dir_path)

# from langchain_community.tools.requests.tool import RequestsGetTool
# requests_tool = RequestsGetTool(allow_dangerous_requests=False)


def get_all_tools():
    return [read_file, list_dir, grep_search, glob, read_dir_tree]

if __name__ == '__main__':
    # from langchain.tools import BaseTool
    # print(isinstance(web_search, BaseTool))
    # for t in get_all_tools():
    #     print(t.args_schema.model_json_schema())
    
    # print(read_file.args_schema.model_json_schema())
    # print(read_dir_tree.)
    root = '/root/workspace/code/RUN/DevEval/agent/gpt/sampling/generate_code/running_env/feedparser.urls.convert_to_idn_6790760d-03e4-42cf-bef6-acfa065c263e'
    
    print(print_directory_tree(root))