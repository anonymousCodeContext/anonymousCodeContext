import os
import javalang
from typing import Dict, Optional, List
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

class JavaCodebaseIndexer:
    """
    A tool to index a Java codebase and retrieve source code of methods and classes.
    Uses both javalang for parsing and tree-sitter for precise code extraction.
    """
    
    def __init__(self):
        """Initialize the indexer with tree-sitter parser for Java."""
        self.parser = Parser()
        self.parser.set_language(get_language('java'))
        self.indexed_files: Dict[str, bytes] = {}
        self.trees: Dict[str, any] = {}  # tree-sitter trees
        self.javalang_trees: Dict[str, javalang.tree.CompilationUnit] = {}  # javalang trees
        self.root_dir: Optional[str] = None
        
    def index(self, codebase_path: str):
        """
        Traverse a directory and index all Java files.
        
        Args:
            codebase_path: Root path of the codebase to index
        """
        # Check if already indexed
        if self.root_dir == codebase_path and self.trees:
            print(f"✅ Using cached index for '{codebase_path}' (files: {len(self.trees)})")
            return
        
        # Clear previous index if switching roots
        if self.root_dir and self.root_dir != codebase_path:
            self.indexed_files.clear()
            self.trees.clear()
            self.javalang_trees.clear()
        
        self.root_dir = codebase_path
        print(f"🚀 Starting Java codebase indexing of '{codebase_path}'...")
        
        indexed_count = 0
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Index with tree-sitter
                        self.indexed_files[file_path] = content
                        self.trees[file_path] = self.parser.parse(content)
                        
                        # Also parse with javalang for high-level structure
                        try:
                            java_tree = javalang.parse.parse(content.decode('utf-8'))
                            self.javalang_trees[file_path] = java_tree
                        except:
                            # Javalang parsing might fail for some files
                            pass
                        
                        indexed_count += 1
                        if indexed_count % 10 == 0:
                            print(f"  - Indexed {indexed_count} files...")
                            
                    except Exception as e:
                        print(f"  - Failed to index {file_path}: {e}")
        
        print(f"✅ Java indexing complete. Indexed {indexed_count} files.")
    
    def ensure_indexed(self, codebase_path: str):
        """Ensure the codebase is indexed."""
        self.index(codebase_path)
    
    def get_method_code(self, file_path: str, method_name: str, class_name: Optional[str] = None) -> Optional[str]:
        """
        Retrieve the full source code of a method.
        
        Args:
            file_path: Path to the Java file
            method_name: Name of the method
            class_name: Name of the containing class (optional)
            
        Returns:
            Source code of the method as string, or None if not found
        """
        tree = self.trees.get(file_path)
        file_content = self.indexed_files.get(file_path)
        
        if not tree or not file_content:
            return None
        
        # Validate identifiers
        if not self._is_valid_identifier(method_name):
            print(f"⚠️  Invalid method name: {method_name}")
            return None
        if class_name and not self._is_valid_identifier(class_name):
            print(f"⚠️  Invalid class name: {class_name}")
            return None
        
        if class_name:
            # Check if this is a constructor request
            if method_name == class_name:
                # Query for constructor
                query_str = f"""
                (class_declaration
                  name: (identifier) @class.name
                  body: (class_body
                    (constructor_declaration) @method.def
                  )
                  (#eq? @class.name "{class_name}")
                )
                """
            else:
                # Query for method inside a class
                query_str = f"""
                (class_declaration
                  name: (identifier) @class.name
                  body: (class_body
                    (method_declaration
                      name: (identifier) @method.name
                      (#eq? @method.name "{method_name}")
                    ) @method.def
                  )
                  (#eq? @class.name "{class_name}")
                )
                """
        else:
            # Query for top-level method (rare in Java, but possible in some contexts)
            query_str = f"""
            (method_declaration
              name: (identifier) @method.name
              (#eq? @method.name "{method_name}")
            ) @method.def
            """
        
        language = get_language('java')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)
        
        for node, name in captures:
            if name == 'method.def':
                return node.text.decode('utf-8')
        
        # If not found with tree-sitter, try with javalang as backup
        return self._get_method_code_javalang(file_path, method_name, class_name)
    
    def _get_method_code_javalang(self, file_path: str, method_name: str, class_name: Optional[str] = None) -> Optional[str]:
        """Backup method to get method code using javalang."""
        java_tree = self.javalang_trees.get(file_path)
        file_content = self.indexed_files.get(file_path)
        
        if not java_tree or not file_content:
            return None
        
        # Search for the class and method
        if java_tree.types:
            for type_decl in java_tree.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration):
                    if not class_name or type_decl.name == class_name:
                        for member in type_decl.body:
                            if isinstance(member, javalang.tree.MethodDeclaration) and member.name == method_name:
                                # Extract method code (approximation)
                                return self._extract_member_code(file_content.decode('utf-8'), member)
                            elif isinstance(member, javalang.tree.ConstructorDeclaration) and method_name == class_name:
                                return self._extract_member_code(file_content.decode('utf-8'), member)
        
        return None
    
    def _extract_member_code(self, source: str, member) -> str:
        """Extract code for a member from source (rough approximation)."""
        # This is a simplified extraction - in production, you'd want more precise extraction
        lines = source.split('\n')
        
        # Try to find method signature
        for i, line in enumerate(lines):
            if hasattr(member, 'name') and member.name in line:
                # Found potential start, now find the end
                brace_count = 0
                start_idx = i
                started = False
                
                for j in range(i, len(lines)):
                    line = lines[j]
                    for char in line:
                        if char == '{':
                            brace_count += 1
                            started = True
                        elif char == '}':
                            brace_count -= 1
                    
                    if started and brace_count == 0:
                        # Found the end
                        return '\n'.join(lines[start_idx:j+1])
        
        return None
    
    def get_class_code(self, file_path: str, class_name: str) -> Optional[str]:
        """
        Retrieve the full source code of a class.
        
        Args:
            file_path: Path to the Java file
            class_name: Name of the class
            
        Returns:
            Source code of the class as string, or None if not found
        """
        tree = self.trees.get(file_path)
        file_content = self.indexed_files.get(file_path)
        
        if not tree or not file_content:
            return None
        
        if not self._is_valid_identifier(class_name):
            print(f"⚠️  Invalid class name: {class_name}")
            return None
        
        query_str = f"""
        (class_declaration
          name: (identifier) @class.name
          (#eq? @class.name "{class_name}")
        ) @class.def
        """
        
        language = get_language('java')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)
        
        for node, name in captures:
            if name == 'class.def':
                return node.text.decode('utf-8')
        
        return None
    
    def get_field_code(self, file_path: str, class_name: str, field_name: str) -> Optional[str]:
        """
        Retrieve the line of code where a field is declared.
        
        Args:
            file_path: Path to the Java file
            class_name: Name of the class
            field_name: Name of the field
            
        Returns:
            Field declaration as string, or None if not found
        """
        tree = self.trees.get(file_path)
        
        if not tree:
            return None
        
        if not self._is_valid_identifier(class_name) or not self._is_valid_identifier(field_name):
            print(f"⚠️  Invalid identifier: class='{class_name}', field='{field_name}'")
            return None
        
        query_str = f"""
        (class_declaration
          name: (identifier) @class.name
          body: (class_body
            (field_declaration
              declarator: (variable_declarator
                name: (identifier) @field.name
                (#eq? @field.name "{field_name}")
              )
            ) @field.def
          )
          (#eq? @class.name "{class_name}")
        )
        """
        
        language = get_language('java')
        query = language.query(query_str)
        captures = query.captures(tree.root_node)
        
        for node, name in captures:
            if name == 'field.def':
                return node.text.decode('utf-8')
        
        return None
    
    def get_static_field_code(self, file_path: str, class_name: str, field_name: str) -> Optional[str]:
        """
        Retrieve static field declaration.
        
        Args:
            file_path: Path to the Java file
            class_name: Name of the class
            field_name: Name of the static field
            
        Returns:
            Static field declaration as string, or None if not found
        """
        # For enum values, handle specially
        if class_name and field_name:
            # Check if it's an enum
            tree = self.trees.get(file_path)
            if not tree:
                return None
            
            # First try to find as enum constant
            query_str = f"""
            (enum_declaration
              name: (identifier) @enum.name
              body: (enum_body
                (enum_constant
                  name: (identifier) @constant.name
                  (#eq? @constant.name "{field_name}")
                ) @constant.def
              )
              (#eq? @enum.name "{class_name}")
            )
            """
            
            try:
                language = get_language('java')
                query = language.query(query_str)
                captures = query.captures(tree.root_node)
                
                for node, name in captures:
                    if name == 'constant.def':
                        return node.text.decode('utf-8')
            except:
                pass
            
            # Try as regular static field
            query_str = f"""
            (class_declaration
              name: (identifier) @class.name
              body: (class_body
                (field_declaration) @field.def
              )
              (#eq? @class.name "{class_name}")
            )
            """
            
            try:
                language = get_language('java')
                query = language.query(query_str)
                captures = query.captures(tree.root_node)
                
                for node, name in captures:
                    if name == 'field.def':
                        field_text = node.text.decode('utf-8')
                        # Check if it contains the field name and is static
                        if field_name in field_text and 'static' in field_text:
                            return field_text
            except:
                pass
        
        return None
    
    def _is_valid_identifier(self, identifier: str) -> bool:
        """Check if a string is a valid Java identifier."""
        if not identifier:
            return False
        
        # Java identifier rules
        if not (identifier[0].isalpha() or identifier[0] == '_' or identifier[0] == '$'):
            return False
        
        for char in identifier[1:]:
            if not (char.isalnum() or char == '_' or char == '$'):
                return False
        
        return True
    
    def get_all_methods_in_file(self, file_path: str) -> List[str]:
        """
        Get all method signatures in a file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            List of method identifiers in format 'ClassName::methodName'
        """
        methods = []
        java_tree = self.javalang_trees.get(file_path)
        
        if not java_tree:
            return methods
        
        if java_tree.types:
            for type_decl in java_tree.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration):
                    class_name = type_decl.name
                    for member in type_decl.body:
                        if isinstance(member, javalang.tree.MethodDeclaration):
                            methods.append(f"{class_name}::{member.name}")
                        elif isinstance(member, javalang.tree.ConstructorDeclaration):
                            methods.append(f"{class_name}::{class_name}")
        
        return methods
