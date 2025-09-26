import os
from typing import Dict, Optional, List
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from gensim.summarization import bm25
from gensim.utils import simple_preprocess
from langchain_core.tools import tool

class NavigationPlus:
    def __init__(self, project_root: str):# project_root: the root directory of the project
        self.project_root = os.path.abspath(project_root)
        self.parser = Parser()
        self.parser.set_language(get_language('python'))
        self.indexed_files: Dict[str, bytes] = {}
        self.trees: Dict[str, any] = {}
        self.bm25_model = None
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

    def index(self):
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
                        corpus_docs.append(simple_preprocess(file_content_str))
                        doc_paths.append(file_path)
                        
                        print(f"  - Indexed: {file_path}")
                    except Exception as e:
                        print(f"  - Failed to index {file_path}: {e}")
        
        if corpus_docs:
            self.bm25_model = bm25.BM25(corpus_docs)
            self.file_paths = doc_paths
            print("✅ BM25 model created.")
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
        if not self.bm25_model:
            return ["BM25 model is not indexed."]

        query_tokens = simple_preprocess(query)
        scores = self.bm25_model.get_scores(query_tokens)

        # Get top_k scores
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_k_indices:
            results.append(f"{self.file_paths[i]} (score: {scores[i]:.4f})")
        return results

def get_all_tools(navigator: NavigationPlus):
    # This function will now be responsible for creating tool instances
    # with the provided navigator instance.
    
    @tool(parse_docstring=True)
    def read_dir_tree(path: str) -> str:
        """
        Provides a recursive view of the directory hierarchy.
        
        Args:
            path: The path of the directory to read.

        Returns:
            The directory tree.
        """
        
        def print_directory_tree(startpath, prefix=''):
            entries = sorted(os.listdir(startpath), key=lambda x: (not os.path.isdir(os.path.join(startpath, x)), x))
            trees = ''
            for i, entry in enumerate(entries):
                if entry in navigator.exclude_dirs or entry.startswith('.'):
                    continue
                is_last = i == len(entries) - 1
                entry_path = os.path.join(startpath, entry)

                connector = '└── ' if is_last else '├── '
                trees += f"{prefix}{connector}{entry}\n"
                if os.path.isdir(entry_path):
                    extension = '    ' if is_last else '│   '
                    trees += print_directory_tree(entry_path, prefix + extension)
            return trees

        full_path = navigator._secure_path(path)
        if not full_path:
            return f"Error: Access denied or invalid path: {path}"
        
        if not os.path.exists(full_path):
            return f"Error: directory does not exist: {path}"
        if not os.path.isdir(full_path):
            return f"Error: {path} is a file, not a directory"
            
        return print_directory_tree(full_path)

    @tool(parse_docstring=True)
    def find_definition(identifier: str) -> str:
        """
        Locates an identifier's definition using the structural index.
        
        Args:
            identifier: The identifier to locate.

        Returns:
            The definition of the identifier.
        """
        return navigator.find_definition(identifier)

    @tool(parse_docstring=True)
    def find_references(identifier: str) -> str:
        """
        Finds all usages of an identifier using the structural index.
        
        Args:
            identifier: The identifier to find references to.

        Returns:
            A list of all found references.
        """
        references = navigator.find_references(identifier)
        return "\n".join(references) if references else "No references found."

    @tool(parse_docstring=True)
    def read_class_structure(class_name: str, file_path: str) -> str:
        """
        Extracts a high-level view of a class's methods and fields.
        
        Args:
            class_name: The name of the class.
            file_path: The path to the file containing the class.

        Returns:
            A string representing the class structure.
        """
        return navigator.read_class_structure(class_name, file_path)

    @tool(parse_docstring=True)
    def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        Reads raw file content, supporting line ranges.
        
        Args:
            file_path: The path of the file to read.
            start_line: The start line of the file to read.
            end_line: The end line of the file to read.

        Returns:
            The content of the file.
        """
        return navigator.read_file(file_path, start_line, end_line)

    @tool(parse_docstring=True)
    def read_function(function_name: str, file_path: str, class_name: Optional[str] = None) -> str:
        """
        Extracts only the source code of a specific function.
        
        Args:
            function_name: The name of the function.
            file_path: The path to the file containing the function.
            class_name: The name of the class if the function is a method.

        Returns:
            The source code of the function.
        """
        return navigator.read_function(function_name, file_path, class_name)

    @tool(parse_docstring=True)
    def read_class(class_name: str, file_path: str) -> str:
        """
        Extracts the full source code of a specific class.
        
        Args:
            class_name: The name of the class.
            file_path: The path to the file containing the class.

        Returns:
            The source code of the class.
        """
        return navigator.read_class(class_name, file_path)

    @tool(parse_docstring=True)
    def semantic_search(query: str, top_k: int = 5) -> str:
        """
        Searches for semantically similar code snippets using a vector index.
        
        Args:
            query: The query to search for.
            top_k: The number of results to return.

        Returns:
            A list of the most similar code snippets.
        """
        results = navigator.semantic_search(query, top_k)
        return "\n".join(results) if results else "No similar results found."

    return [
        read_dir_tree,
        find_definition,
        find_references,
        read_class_structure,
        read_file,
        read_function,
        read_class,
        semantic_search,
    ]
