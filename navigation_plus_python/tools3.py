from langchain_core.tools import tool
import os
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from navigation.agentic_coder import AgentHelper
import glob as glob_module
import os
from typing import Dict, Optional, List
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from gensim.summarization import bm25
from gensim.utils import simple_preprocess
from langchain_core.tools import tool
import numpy as np
import torch
import torch.nn.functional as F
from gen_unixcoder import Unixcoder_Embedding


class NavigationPlus:
    def __init__(self, project_root: str):# project_root: the root directory of the project
        self.project_root = os.path.abspath(project_root)
        self.parser = Parser()
        self.parser.set_language(get_language('python'))
        self.indexed_files: Dict[str, bytes] = {}
        self.trees: Dict[str, any] = {}
        # self.bm25_model = None
        # self.semantic_model = None
        self.corpus_embs = None
        self.file_indeics = None
        self.file_paths: List[str] = []
        # Common directories to exclude from indexing and directory listing
        self.exclude_dirs = {'venv', 'env', '.git', '__pycache__', 'tests', 'myenv', 'dist', 'build', '.vscode', '.idea'}

    def _secure_path(self, file_path: str) -> Optional[str]:
        """
        Resolves a file path to an absolute path, ensuring it's within the project root.
        """
        # Prevent directory traversal attacks
        if ".." in file_path:
            return None
        
        full_path = os.path.abspath(os.path.join(self.project_root, file_path))
        
        if os.path.commonpath([self.project_root]) == os.path.commonpath([self.project_root, full_path]):
            return full_path
        return None

    def index(self, corpus_embs, file_indeics):
        """
        Indexes the entire codebase, including structural and semantic indexing.
        """
        print(f"🚀 Starting indexing of '{self.project_root}'...")
        corpus_docs = []
        doc_paths = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not d.startswith('.')]
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        self.indexed_files[file_path] = content
                        tree = self.parser.parse(content)
                        self.trees[file_path] = tree
                        
                        # For BM25, we need a list of lists of tokens
                        file_content_str = content.decode('utf-8', errors='ignore')
                        # corpus_docs.append(simple_preprocess(file_content_str))
                        doc_paths.append(file_path)
                        
                        print(f"  - Indexed: {file_path}")
                    except Exception as e:
                        print(f"  - Failed to index {file_path}: {e}")
        
        # if corpus_docs:
        #     self.bm25_model = bm25.BM25(corpus_docs)
        #     self.file_paths = doc_paths
        #     print("✅ BM25 model created.")
        self.corpus_embs = corpus_embs
        self.file_indeics = file_indeics
        print("✅ Indexing complete.")

    def find_definition(self, identifier: str, file_path: Optional[str] = None) -> Optional[str]:
        """
        Finds the definition of an identifier.
        """
        # For now, this is a placeholder. A real implementation would use the index
        # to find the definition of the identifier.
        # It would look through the indexed files and use the ASTs to find where
        # the identifier is defined.
        
        # A simplified example:
        for path, tree in self.trees.items():
            # This is a very basic search. A real implementation would be more sophisticated.
            query_str = f"""
            (function_definition
              name: (identifier) @function.name
              (#eq? @function.name "{identifier}")
            ) @function.def
            """
            language = get_language('python')
            query = language.query(query_str)
            captures = query.captures(tree.root_node)
            for node, name in captures:
                if name == 'function.def':
                    return f"Definition found in {path}:\n{node.text.decode('utf-8')}"
        return "Definition not found."

    def find_references(self, identifier: str) -> List[str]:
        """
        Finds all references to an identifier.
        """
        references = []
        for path, tree in self.trees.items():
            query_str = f"""
            (identifier) @id
            (#eq? @id "{identifier}")
            """
            language = get_language('python')
            query = language.query(query_str)
            captures = query.captures(tree.root_node)
            for node, name in captures:
                if name == 'id':
                    line_num = node.start_point[0] + 1
                    line_content = self.indexed_files[path].splitlines()[line_num - 1].decode('utf-8', errors='ignore')
                    references.append(f"{path}, line {line_num}: {line_content.strip()}")
        return references

    def read_class_structure(self, class_name: str, file_path: str) -> Optional[str]:
        """
        Reads the structure of a class, including its methods and fields.
        """
        tree = self.trees.get(file_path)
        if not tree:
            return "File not found in index."

        query_str = f"""
        (class_definition
          name: (identifier) @class.name
          body: (block) @class.body
          (#eq? @class.name "{class_name}")
        )
        """
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        if not captures:
            return f"Class '{class_name}' not found in '{file_path}'."

        class_node = captures[0][0]
        structure = f"class {class_name}:\n"
        
        body_node = None
        for child in class_node.children:
            if child.type == 'block':
                body_node = child
                break
        
        if body_node:
            for node in body_node.children:
                if node.type == 'function_definition':
                    func_name_node = node.child_by_field_name('name')
                    if func_name_node:
                        structure += f"    def {func_name_node.text.decode('utf-8', errors='ignore')}(...)\n"
                elif node.type == 'expression_statement' and node.children[0].type == 'assignment':
                    assignment_node = node.children[0]
                    left_node = assignment_node.child_by_field_name('left')
                    if left_node and left_node.type == 'identifier':
                         structure += f"    {left_node.text.decode('utf-8', errors='ignore')}\n"

        return structure

    def read_file(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        Reads the content of a file, with optional line range.
        """
        full_path = self._secure_path(file_path)
        if not full_path:
            return f"Error: Access denied or invalid path: {file_path}"
        
        if not os.path.exists(full_path):
            return f"Error: File does not exist: {file_path}"
        if os.path.isdir(full_path):
            return f"Error: {file_path} is a directory, not a file"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if start_line and end_line:
                    return "".join(lines[start_line-1:end_line])
                elif start_line:
                    return "".join(lines[start_line-1:])
                elif end_line:
                    return "".join(lines[:end_line])
                else:
                    return "".join(lines)
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

    def read_function(self, function_name: str, file_path: str, class_name: Optional[str] = None) -> str:
        """
        Extracts the source code of a specific function.
        """
        secure_path = self._secure_path(file_path)
        if not secure_path:
            return f"Error: Access denied or invalid path: {file_path}"
        
        tree = self.trees.get(secure_path)
        if not tree:
            return "File not found in index."

        if class_name:
            query_str = f"""
            (class_definition
              name: (identifier) @class.name
              body: (block
                (function_definition
                  name: (identifier) @function.name
                  (#eq? @function.name "{function_name}")
                ) @function.def
              )
              (#eq? @class.name "{class_name}")
            )
            """
        else:
            query_str = f"""
            (function_definition
              name: (identifier) @function.name
              (#eq? @function.name "{function_name}")
            ) @function.def
            """
        
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name == 'function.def':
                return node.text.decode('utf-8', errors='ignore')

        return f"Function '{function_name}' not found in '{file_path}'."

    def read_class(self, class_name: str, file_path: str) -> str:
        """
        Extracts the full source code of a specific class.
        """
        secure_path = self._secure_path(file_path)
        if not secure_path:
            return f"Error: Access denied or invalid path: {file_path}"
            
        tree = self.trees.get(secure_path)
        if not tree:
            return "File not found in index."

        query_str = f"""
        (class_definition
          name: (identifier) @class.name
          (#eq? @class.name "{class_name}")
        ) @class.def
        """
        
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name == 'class.def':
                return node.text.decode('utf-8', errors='ignore')

        return f"Class '{class_name}' not found in '{file_path}'."

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Searches for semantically similar code snippets using a BM25 index.
        """
        if not self.file_indeics:
            return ["embedding model is not indexed."]

        # query_tokens = simple_preprocess(query)
        # average_idf = sum(float(val) for val in self.bm25_model.idf.values()) / len(self.bm25_model.idf)
        # scores = self.bm25_model.get_scores(query_tokens, average_idf)

        # # Get top_k scores
        # top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # results = []
        # for i in top_k_indices:
        #     results.append(f"{self.file_paths[i]} (score: {scores[i]:.4f})")
            
        query = Unixcoder_Embedding().embedding(query)
        query = torch.tensor(query).unsqueeze(0)
        scores = F.cosine_similarity(query, self.corpus_embs, dim=1).numpy()
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_k_indices:
            results.append(f"{self.file_indeics[i]} (score: {scores[i]:.4f})")
            
        return results


# Proxy Pattern
class ToolPlusImpl:
    ALL_TOOLS = dict()
    PROJ_FILE_EMBEDDINGS_PATH = 'data/DevEval/similarity/proj_file_embedding'
    @classmethod
    def get_tools_instance(cls, task_id: str):
        """
        Factory method to create ToolImpl instances.
        If an instance with the same task_id already exists, it returns the existing instance; otherwise, it creates and returns a new one.
        """
        if task_id not in cls.ALL_TOOLS:
            meta_data = AgentHelper.STATES[task_id]
            instance = ToolPlusImpl(task_id, meta_data)
            ToolPlusImpl.ALL_TOOLS[task_id] = instance
        instance = cls.ALL_TOOLS[task_id]
        return instance
    
    def __init__(self, task_id: str, meta_data: dict):
        self.task_id = task_id
        self.ALL_TOOLS[task_id] = self
        run_proj_path = AgentHelper.STATES[task_id]['project_path']
        self.project_path = run_proj_path
        self.navigator = NavigationPlus(run_proj_path)
        
        meta_data = AgentHelper.STATES[task_id]['meta_data']
        pj_path = meta_data['project_path']
        embeddings_dir = os.path.join(self.PROJ_FILE_EMBEDDINGS_PATH, pj_path)
        embeddings_file = os.path.join(embeddings_dir, 'proj_file_embeddings.npy')
        corpus_embs = np.load(embeddings_file)
        file_indeics = []
        with open(os.path.join(embeddings_dir, 'files_path.txt'), 'r') as f:
            for line in f:
                file_indeics.append(line.strip())        
        corpus_embs = torch.tensor(corpus_embs)

        self.navigator.index(corpus_embs, file_indeics)
    
    def get_navigator(self):
        return self.navigator
    
    def read_file(self, file_path: str, start_line: int=None, end_line: int=None) -> str:
        proj_path = self.project_path
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

    def list_dir(self, dir_path: str) -> list[str]:
        proj_path = self.project_path
        a_dir_path = os.path.join(proj_path, dir_path)
        if os.path.exists(a_dir_path) is False:
            return f"Error: directory does not exist: {dir_path}"
        elif os.path.isfile(a_dir_path):
            return f"Error: {dir_path} is a file, not a directory"
        
        return os.listdir(a_dir_path)
    
    def grep_search(self, file_path: str, pattern: str) -> str:
        proj_path = self.project_path
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
        
    def glob(self, dir_path: str, pattern: str) -> list[str]:
        proj_path = self.project_path
        a_dir_path = os.path.join(proj_path, dir_path)
        if os.path.exists(a_dir_path) is False:
            return f"Error: directory does not exist: {dir_path}"
        elif os.path.isfile(a_dir_path):
            return f"Error: {dir_path} is a file, not a directory"
        
        search_pattern = os.path.join(a_dir_path, pattern)
        return glob_module.glob(search_pattern, recursive=True)
    
    def read_dir_tree(self, path: str) -> str:
        proj_path = self.project_path
        a_dir_path = os.path.join(proj_path, path)
        if os.path.exists(a_dir_path) is False:
            return f"Error: directory does not exist: {path}"
        elif os.path.isfile(a_dir_path):
            return f"Error: {path} is a file, not a directory"
        
        return print_directory_tree(a_dir_path)

    def finish(self):
        del self.ALL_TOOLS[self.task_id]

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
    return ToolPlusImpl.get_tools_instance(task_id).read_file(file_path, start_line, end_line)
    
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
    return ToolPlusImpl.get_tools_instance(task_id).list_dir(dir_path)

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
    return ToolPlusImpl.get_tools_instance(task_id).grep_search(file_path, pattern)

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
    return ToolPlusImpl.get_tools_instance(task_id).glob(dir_path, pattern)


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
    return ToolPlusImpl.get_tools_instance(task_id).read_dir_tree(path)

# from langchain_community.tools.requests.tool import RequestsGetTool
# requests_tool = RequestsGetTool(allow_dangerous_requests=False)

@tool(parse_docstring=True)
def find_definition(task_id: str, identifier: str) -> str:
    """
    Locates an identifier's definition using the structural index.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        identifier: The identifier to locate.

    Returns:
        The definition of the identifier.
    """
    # return navigator.find_definition(identifier)
    return ToolPlusImpl.get_tools_instance(task_id).get_navigator().find_definition(identifier)

@tool(parse_docstring=True)
def find_references(task_id: str, identifier: str) -> str:
    """
    Finds all usages of an identifier using the structural index.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        identifier: The identifier to find references to.

    Returns:
        A list of all found references.
    """
    references = ToolPlusImpl.get_tools_instance(task_id).get_navigator().find_references(identifier)
    return "\n".join(references) if references else "No references found."
    
@tool(parse_docstring=True)
def read_class_structure(task_id: str, class_name: str, file_path: str) -> str:
    """
    Extracts a high-level view of a class's methods and fields.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        class_name: The name of the class.
        file_path: The path to the file containing the class.

    Returns:
        A string representing the class structure.
    """
    return ToolPlusImpl.get_tools_instance(task_id).get_navigator().read_class_structure(class_name, file_path)

@tool(parse_docstring=True)
def read_function(task_id: str, function_name: str, file_path: str, class_name: Optional[str] = None) -> str:
    """
    Extracts only the source code of a specific function.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        function_name: The name of the function.
        file_path: The path to the file containing the function.
        class_name: The name of the class if the function is a method.

    Returns:
        The source code of the function.
    """
    return ToolPlusImpl.get_tools_instance(task_id).get_navigator().read_function(function_name, file_path, class_name)


@tool(parse_docstring=True)
def read_class(task_id: str, class_name: str, file_path: str) -> str:
    """
    Extracts the full source code of a specific class.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        class_name: The name of the class.
        file_path: The path to the file containing the class.

    Returns:
        The source code of the class.
    """
    return ToolPlusImpl.get_tools_instance(task_id).get_navigator().read_class(class_name, file_path)

@tool(parse_docstring=True)
def semantic_search(task_id: str, query: str, top_k: int = 5) -> str:
    """
    Perform semantic search on the code repository and return: the paths and corresponding scores of the top_k files most semantically relevant to the query in the code repository. Note that the search granularity is at the file level, not within file contents.

    For example, query='how to connect to a database?' top_k=2, return: ['mssqlcli/mssqlcliclient.py (score: 0.2698)', 'mssqlcli/packages/special/commands.py (score: 0.1680)'], where score indicates semantic similarity score, with higher values indicating greater relevance.

    A bad example: query='get_tools function in utils/tools.py', as this tool does not support searching within file contents.

    Args:
        task_id: your task_id, which is a required parameter for all the provided tools.
        query: The query to search for.
        top_k: The number of results to return.

    Returns:
        A list of the most similar code snippets.
    """
    results = ToolPlusImpl.get_tools_instance(task_id).get_navigator().semantic_search(query, top_k)
    return "\n".join(results) if results else "No similar results found."

def get_all_tools():
    # return [read_file, list_dir, grep_search, glob, read_dir_tree]
    # return [read_file, 
    #         read_dir_tree, 
    #         find_definition, 
    #         # find_references, 
    #         read_class_structure, 
    #         read_function,
    #         read_class,
    #         semantic_search]
    return [read_file, list_dir, grep_search, glob, read_dir_tree, find_definition, read_class_structure, read_function, read_class, semantic_search]


    
    
if __name__ == "__main__":
    import json
    for t in get_all_tools():
        print(json.dumps(t.args_schema.model_json_schema(), indent=4))