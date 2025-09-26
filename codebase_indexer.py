import os
from typing import Dict, Optional, List
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

class CodebaseIndexer:
    """
    A tool to index a Python codebase and retrieve the full source code of functions.
    """

    def __init__(self):
        """Initializes the indexer with a tree-sitter parser for Python."""
        self.parser = Parser()
        self.parser.set_language(get_language('python'))
        self.indexed_files: Dict[str, bytes] = {}
        self.trees: Dict[str, any] = {}
        self.root_dir: Optional[str] = None

    def index(self, codebase_path: str):
        """
        Traverses a directory, reads all Python files, and parses them into ASTs.
        
        :param codebase_path: The root path of the codebase to index.
        """
        # If already indexed for this root, skip for performance
        if self.root_dir == codebase_path and self.trees:
            print(f"✅ Using cached index for '{codebase_path}' (files: {len(self.trees)})")
            return

        # If switching roots, clear previous state
        if self.root_dir and self.root_dir != codebase_path:
            self.indexed_files.clear()
            self.trees.clear()

        self.root_dir = codebase_path
        print(f"🚀 Starting indexing of '{codebase_path}'...")
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        self.indexed_files[file_path] = content
                        self.trees[file_path] = self.parser.parse(content)
                        print(f"  - Indexed: {file_path}")
                    except Exception as e:
                        print(f"  - Failed to index {file_path}: {e}")
        print("✅ Indexing complete.")

    def ensure_indexed(self, codebase_path: str):
        """Index the codebase if not already indexed for the given root."""
        self.index(codebase_path)

    def get_function_code(self, file_path: str, func_name: str, class_name: Optional[str] = None) -> Optional[str]:
        """
        Retrieves the full source code of a function/method using its location.

        :param file_path: The path to the file containing the function.
        :param func_name: The name of the function.
        :param class_name: The name of the class if the function is a method.
        :return: The full source code of the function as a string, or None if not found.
        """
        if not func_name.isidentifier() or (class_name and not class_name.isidentifier()):
            print(f"⚠️  Skipping query due to invalid identifier: class='{class_name}', func='{func_name}'")
            return None

        tree = self.trees.get(file_path)
        file_content = self.indexed_files.get(file_path)
        if not tree or not file_content:
            return None

        if class_name:
            # Query for a method inside a class (support decorated methods too)
            query_str = f"""
            (class_definition
              name: (identifier) @class.name
              body: (block
                [
                  (function_definition
                    name: (identifier) @function.name) @function.def
                  (decorated_definition
                    (function_definition
                      name: (identifier) @function.name) @function.def)
                ]
              )
              (#eq? @class.name "{class_name}")
              (#eq? @function.name "{func_name}")
            )
            """
        else:
            # Query for a top-level function (support decorated definitions)
            query_str = f"""
            [
              (function_definition
                name: (identifier) @function.name
                (#eq? @function.name "{func_name}")
              ) @function.def
              (decorated_definition
                (function_definition
                  name: (identifier) @function.name
                  (#eq? @function.name "{func_name}")
                ) @function.def
              )
            ]
            """
        
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name == 'function.def':
                return node.text.decode('utf-8')
        
        return None

    def get_class_code(self, file_path: str, class_name: str) -> Optional[str]:
        """
        Retrieves the full source code of a class.

        :param file_path: The path to the file containing the class.
        :param class_name: The name of the class.
        :return: The full source code of the class as a string, or None if not found.
        """
        if not class_name.isidentifier():
            print(f"⚠️  Skipping query due to invalid identifier: class='{class_name}'")
            return None
        tree = self.trees.get(file_path)
        file_content = self.indexed_files.get(file_path)
        if not tree or not file_content:
            return None

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
                return node.text.decode('utf-8')
        
        return None

    def get_attribute_code(self, file_path: str, class_name: str, attribute_name: str) -> Optional[str]:
        """
        Retrieves the line of code where an attribute is defined in the __init__ method.

        :param file_path: The path to the file containing the class.
        :param class_name: The name of the class.
        :param attribute_name: The name of the attribute.
        :return: The line of code as a string, or None if not found.
        """
        if not class_name.isidentifier() or not attribute_name.isidentifier():
             print(f"⚠️  Skipping query due to invalid identifier: class='{class_name}', attribute='{attribute_name}'")
             return None

        tree = self.trees.get(file_path)
        if not tree:
            return None

        query_str = f"""
        (class_definition
          name: (identifier) @class.name
          body: (block
            (function_definition
              name: (identifier) @init.name
              body: (block
                (expression_statement
                  (assignment
                    left: (attribute
                      object: (identifier) @self
                      attribute: (identifier) @attr.name
                      (#eq? @self "self")
                      (#eq? @attr.name "{attribute_name}"))
                  ) @assignment
                )
              )
              (#eq? @init.name "__init__")
            )
          )
          (#eq? @class.name "{class_name}")
        )
        """
        
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name == 'assignment':
                return node.text.decode('utf-8')
        
        return None

    def get_module_variable_code(self, file_path: str, variable_name: str) -> Optional[str]:
        """
        Retrieves the line of code where a module-level variable is defined.

        :param file_path: The path to the file containing the variable.
        :param variable_name: The name of the variable.
        :return: The line of code as a string, or None if not found.
        """
        if not variable_name.isidentifier():
             print(f"⚠️  Skipping query due to invalid identifier: variable='{variable_name}'")
             return None

        tree = self.trees.get(file_path)
        if not tree:
            return None

        query_str = f"""
        (expression_statement
          (assignment
            left: (identifier) @var.name
            (#eq? @var.name "{variable_name}")
          ) @assignment
        )
        """
        
        language = get_language('python')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name == 'assignment':
                # We need to check if this is a top-level assignment
                # A simple check is to see if its parent is the module itself
                if node.parent.parent.type == 'module':
                    return node.text.decode('utf-8')
        
        return None

if __name__ == '__main__':
    # 1. Initialize and index the codebase
    indexer = CodebaseIndexer()
    indexer.index('demo_project') # Assuming 'demo_project' is a valid path

    print("\n--- Test Cases ---\n")

    # 2. Test case: Retrieve a method from a class
    print("1. Testing retrieval of a class method...")
    file_to_query = 'demo_project/imapclient.py'
    class_to_query = 'IMAPClient'
    func_to_query = '_command_and_check'
    
    print(f"   Query: {file_to_query}::{class_to_query}::{func_to_query}")
    
    retrieved_code = indexer.get_function_code(
        file_path=file_to_query,
        class_name=class_to_query,
        func_name=func_to_query
    )

    if retrieved_code:
        print("   ✅ Found Function Code:")
        print("--------------------")
        print(retrieved_code)
        print("--------------------")
    else:
        print("   ❌ Function not found.")

    # 3. Test case: Retrieve a standalone function (hypothetical)
    # To make this work, we would need a file with a standalone function.
    # For now, this will likely fail, which is expected.
    print("\n2. Testing retrieval of a standalone function...")
    file_to_query_standalone = 'demo_project/response_parser.py'
    func_to_query_standalone = 'parse_response'
    
    print(f"   Query: {file_to_query_standalone}::{func_to_query_standalone}")

    retrieved_code_standalone = indexer.get_function_code(
        file_path=file_to_query_standalone,
        func_name=func_to_query_standalone
    )

    if retrieved_code_standalone:
        print("   ✅ Found Function Code:")
        print("--------------------")
        print(retrieved_code_standalone)
        print("--------------------")
    else:
        print("   ❌ Function not found.")

    # 4. Test case: Retrieve a full class
    print("\n3. Testing retrieval of a full class...")
    class_to_query_full = 'IMAPClient'
    
    print(f"   Query: {file_to_query}::{class_to_query_full}")

    retrieved_class_code = indexer.get_class_code(
        file_path=file_to_query,
        class_name=class_to_query_full
    )

    if retrieved_class_code:
        print("   ✅ Found Class Code:")
        print("--------------------")
        print(retrieved_class_code)
        print("--------------------")
    else:
        print("   ❌ Class not found.")
