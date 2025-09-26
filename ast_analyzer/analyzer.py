import ast
import os
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field

from .visitor import FunctionVisitor
from .models import FunctionAnalysisResult, Dependency

@dataclass
class ClassInfo:
    """Holds information about a class, including its methods and parents."""
    name: str
    file_path: str
    methods: Set[str] = field(default_factory=set)
    properties: Set[str] = field(default_factory=set)
    # The fully qualified names of parent classes
    parents: List[str] = field(default_factory=list)
    inherited_methods: Dict[str, str] = field(default_factory=dict) # method_name -> fq_class_name
    inherited_properties: Dict[str, str] = field(default_factory=dict) # property_name -> fq_class_name

class AstAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self._tree_cache: Dict[str, ast.AST] = {}
        self._import_map_cache: Dict[str, Dict[str, str]] = {}
        self._class_info_cache: Dict[str, ClassInfo] = {}

    def _get_tree(self, file_path: str) -> ast.AST:
        if file_path in self._tree_cache:
            return self._tree_cache[file_path]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code, filename=file_path)
        self._tree_cache[file_path] = tree
        return tree

    def _get_module_path(self, file_path: str) -> str:
        rel_path = os.path.relpath(file_path, self.project_root)
        module_path, _ = os.path.splitext(rel_path)
        return module_path.replace(os.sep, '.')
    
    def _parse_imports(self, tree: ast.AST, file_path: str) -> Dict[str, str]:
        if file_path in self._import_map_cache:
            return self._import_map_cache[file_path]

        imports = {}
        current_module = self._get_module_path(file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                
                # Handle relative imports
                if node.level > 0:
                    level = node.level
                    # For a level > 1, we need to go up the package hierarchy
                    package_path = current_module.split('.')
                    if len(package_path) < level:
                         # This is an invalid relative import, but we'll be lenient
                         base_module = ""
                    else:
                         base_module = ".".join(package_path[:-level])

                    if module_name:
                         full_module_name = f"{base_module}.{module_name}" if base_module else module_name
                    else: # from . import ...
                         full_module_name = base_module
                else:
                    full_module_name = module_name

                for alias in node.names:
                    imports[alias.asname or alias.name] = full_module_name

        self._import_map_cache[file_path] = imports
        return imports
    
    def _find_class_node(self, tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _get_class_info(self, class_name: str, file_path: str) -> Optional[ClassInfo]:
        fq_class_name = f"{self._get_module_path(file_path)}.{class_name}"
        if fq_class_name in self._class_info_cache:
            return self._class_info_cache[fq_class_name]

        tree = self._get_tree(file_path)
        imports = self._parse_imports(tree, file_path)
        class_node = self._find_class_node(tree, class_name)

        if not class_node:
            return None

        own_methods: Set[str] = set()
        own_properties: Set[str] = set()
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                is_property = any(
                    isinstance(d, ast.Name) and d.id == 'property'
                    for d in node.decorator_list
                )
                if is_property:
                    own_properties.add(node.name)
                else:
                    own_methods.add(node.name)

        parent_class_names = []
        inherited_methods: Dict[str, str] = {}
        inherited_properties: Dict[str, str] = {}
        for base in class_node.bases:
            parent_fq_name = self._resolve_parent_class_fq_name(base, imports)
            if parent_fq_name:
                parent_class_names.append(parent_fq_name)
                # We need the file path for the parent to get its info
                parent_module_path = ".".join(parent_fq_name.split('.')[:-1])
                parent_file_path = os.path.join(self.project_root, parent_module_path.replace('.', os.sep) + '.py')
                
                if os.path.exists(parent_file_path):
                    parent_info = self._get_class_info(parent_fq_name.split('.')[-1], parent_file_path)
                    if parent_info:
                        # Add direct methods and properties of parent
                        for method in parent_info.methods:
                            inherited_methods[method] = parent_fq_name
                        for prop in parent_info.properties:
                            inherited_properties[prop] = parent_fq_name
                        # Add methods and properties inherited by parent
                        inherited_methods.update(parent_info.inherited_methods)
                        inherited_properties.update(parent_info.inherited_properties)

        class_info = ClassInfo(
            name=fq_class_name,
            file_path=file_path,
            methods=own_methods,
            properties=own_properties,
            parents=parent_class_names,
            inherited_methods=inherited_methods,
            inherited_properties=inherited_properties,
        )
        self._class_info_cache[fq_class_name] = class_info
        return class_info

    def _resolve_parent_class_fq_name(self, base_node: ast.expr, imports: Dict[str, str]) -> Optional[str]:
        """Resolves the fully qualified name of a parent class from its AST node."""
        if isinstance(base_node, ast.Name):
            name = base_node.id
            if name in imports:
                # It's an imported class
                return f"{imports[name]}.{name}"
            else:
                # Assume it's in the same module for now. This is a simplification.
                # A full-fledged resolver would check file scope.
                return name 
        elif isinstance(base_node, ast.Attribute):
            # e.g., parent is something like `mrjob.fs.base.Filesystem`
            # We need to reconstruct the full name
            parts = []
            curr = base_node
            while isinstance(curr, ast.Attribute):
                parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.insert(0, curr.id)
                return ".".join(parts)
        return None

    def _analyze_function_or_method(self, file_path: str, func_name: str, class_name: Optional[str] = None) -> FunctionAnalysisResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        tree = self._get_tree(file_path)
        imports = self._parse_imports(tree, file_path)
        
        class_info = None
        if class_name:
            class_info = self._get_class_info(class_name, file_path)
            if not class_info:
                raise ValueError(f"Could not find class info for {class_name} in {file_path}")

        visitor = FunctionVisitor(target_func_name=func_name, target_class_name=class_name)
        visitor.visit(tree)

        function_id = f"{self._get_module_path(file_path)}::{class_name}::{func_name}" if class_name else f"{self._get_module_path(file_path)}::{func_name}"
        result = FunctionAnalysisResult(function_id=function_id)
        current_module = self._get_module_path(file_path)
        
        for event in visitor.dependencies:
            dep_type = "unknown"
            dep_name = "unknown"
            node = event['node']

            if event['type'] == 'method_call':
                obj_name = event['object_name']
                method_name = event['method_name']
                full_chain = event.get('full_attr_chain') or []
                # calls like self.method(...)
                if obj_name == 'self' and class_info:
                    if method_name in class_info.inherited_methods:
                        dep_type = 'cross_file'
                        dep_name = f"{class_info.inherited_methods[method_name]}.{method_name}"
                    elif method_name in class_info.methods:
                        dep_type = 'intra_class'
                        dep_name = f"{current_module}.{class_name}.{method_name}"
                # calls like module.func(...) imported via alias
                elif obj_name in imports and method_name:
                    dep_type = 'cross_file'
                    dep_name = f"{imports[obj_name]}.{method_name}"
                # chained attribute calls like self.fs.hadoop.get_hadoop_bin(...)
                elif obj_name == 'self' and class_info and full_chain:
                    # Try to map the first attribute after self to a property/attribute
                    # e.g., ['self','fs','hadoop','get_hadoop_bin']
                    first_attr = full_chain[1] if len(full_chain) > 1 else None
                    if first_attr and (first_attr in class_info.properties or first_attr in class_info.inherited_properties):
                        # We don't know the exact class of the attribute statically here,
                        # but we can at least record the property access to trigger pulling
                        # its definition; later, retrieval heuristics can attempt to pull
                        # the guessed class implementations.
                        dep_type = 'attribute_access'
                        dep_name = f"{current_module}.{class_name}.{first_attr}"
            
            elif event['type'] == 'attribute_access':
                obj_name = event['object_name']
                attr_name = event['attribute_name']
                # Default to no knowledge when not analyzing a class
                is_a_method = False
                is_a_property = False
                if class_info:
                    is_a_method = (
                        attr_name in class_info.methods or
                        attr_name in class_info.inherited_methods
                    )
                    is_a_property = (
                        attr_name in class_info.properties or
                        attr_name in class_info.inherited_properties
                    )

                if obj_name == 'self' and class_info:
                    if is_a_property:
                        dep_type = 'property' # Use a distinct type for properties
                        if attr_name in class_info.inherited_properties:
                            dep_type = 'cross_file_property'
                            dep_name = f"{class_info.inherited_properties[attr_name]}.{attr_name}"
                        else:
                            dep_name = f"{current_module}.{class_name}.{attr_name}"
                    elif not is_a_method:
                        # direct attribute access that is not a method/property
                        dep_type = 'attribute_access'
                        dep_name = f"{current_module}.{class_name}.{attr_name}"
            
            elif event['type'] == 'call':
                func_name_called = event['func_name']
                # Check if it's an imported function or class
                if func_name_called in imports:
                    # It could be a function call or a class instantiation
                    # We will treat them as 'cross_file' dependencies for now.
                    # A more sophisticated check could distinguish them.
                    dep_type = 'cross_file'
                    dep_name = f"{imports[func_name_called]}.{func_name_called}"
                else:
                    # Could be an intra-file function call
                    is_in_file = False
                    for top_level_node in tree.body:
                        if isinstance(top_level_node, ast.FunctionDef) and top_level_node.name == func_name_called:
                            is_in_file = True
                            break
                        # Also check for classes defined in the same file
                        if isinstance(top_level_node, ast.ClassDef) and top_level_node.name == func_name_called:
                            is_in_file = True
                            break
                    if is_in_file:
                        dep_type = 'intra_file'
                        dep_name = f"{current_module}.{func_name_called}"

            elif event['type'] == 'module_variable_access':
                var_name = event['variable_name']
                # Ignore built-ins and already imported modules/functions
                try:
                    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
                except Exception:
                    builtins_dict = {}
                # Skip uppercase constants (likely constants) and obvious loop/comp targets
                is_probably_constant = var_name.isupper()
                if (var_name not in builtins_dict and var_name not in imports and not is_probably_constant):
                    # Only treat as module variable if actually defined at module level
                    module_dep = self.find_module_variable_definition(current_module, var_name)
                    if module_dep:
                        dep_type = 'module_variable'
                        dep_name = module_dep.name


            if dep_type != "unknown":
                result.dependencies.add(
                    Dependency(
                        name=dep_name,
                        dependency_type=dep_type,
                        file_path=file_path,
                        line_no=node.lineno
                    )
                )
        return result

    def analyze_function(self, function_id: str) -> FunctionAnalysisResult:
        parts = function_id.split('::')
        
        if len(parts) == 3:
            file_path_str, class_name, func_name = parts
        elif len(parts) == 2:
            file_path_str, func_name = parts
            class_name = None
        else:
            raise ValueError(f"Invalid function_id format: {function_id}. Expected 'path/to/file.py::ClassName::func' or 'path/to/file.py::func'.")

        file_path = os.path.join(self.project_root, file_path_str)
        
        return self._analyze_function_or_method(file_path, func_name, class_name)

    def find_attribute_definition(self, class_name: str, attribute_name: str) -> Optional[Dependency]:
        """
        Finds where a class attribute is defined (e.g., self.fs = ...).
        Searches the class and its parent classes.

        Args:
            class_name: The fully qualified name of the class (e.g., 'mrjob.runner.MRJobRunner').
            attribute_name: The name of the attribute (e.g., 'fs').

        Returns:
            A Dependency object pointing to the assignment line, or None if not found.
        """
        # This is a simplified lookup. A real implementation would handle module resolution better.
        parts = class_name.split('.')
        simple_class_name = parts[-1]
        module_path = ".".join(parts[:-1])
        file_path = os.path.join(self.project_root, module_path.replace('.', os.sep) + '.py')

        if not os.path.exists(file_path):
            # Try to find the file in one of the subdirectories if the module path is not perfect
            for root, _, files in os.walk(self.project_root):
                if f"{simple_class_name}.py" in files: # A heuristic
                    file_path = os.path.join(root, f"{simple_class_name}.py")
                    break
        
        if not os.path.exists(file_path):
            # print(f"Could not find file for class {class_name}")
            return None

        # Start search in the given class, then go to parents
        return self._find_attribute_in_class_hierarchy(class_name, attribute_name, file_path)

    def _find_attribute_in_class_hierarchy(self, fq_class_name: str, attribute_name: str, file_path: str) -> Optional[Dependency]:
        """Recursively search for an attribute definition in a class and its parents."""
        
        simple_class_name = fq_class_name.split('.')[-1]
        
        tree = self._get_tree(file_path)
        # We want to find `self.attribute = ...` inside `__init__`
        visitor = FunctionVisitor(target_class_name=simple_class_name, target_func_name='__init__')
        visitor.visit(tree)

        for event in visitor.dependencies:
            if event['type'] == 'attribute_definition' and event['attribute_name'] == attribute_name:
                return Dependency(
                    name=f"{fq_class_name}.{attribute_name}",
                    dependency_type='attribute_definition',
                    file_path=file_path,
                    line_no=event['node'].lineno
                )

        # If not found, check parent classes
        class_info = self._get_class_info(simple_class_name, file_path)
        if class_info:
            for parent_fq_name in class_info.parents:
                parent_module = ".".join(parent_fq_name.split('.')[:-1])
                parent_file = os.path.join(self.project_root, parent_module.replace('.', os.sep) + '.py')
                if os.path.exists(parent_file):
                    found_dep = self._find_attribute_in_class_hierarchy(parent_fq_name, attribute_name, parent_file)
                    if found_dep:
                        return found_dep
        
        return None

    def find_module_variable_definition(self, module_name: str, variable_name: str) -> Optional[Dependency]:
        """
        Finds where a module-level variable is defined.
        
        Args:
            module_name: The fully qualified name of the module (e.g., 'mrjob.hadoop').
            variable_name: The name of the variable (e.g., 'log').

        Returns:
            A Dependency object pointing to the assignment line, or None if not found.
        """
        file_path = os.path.join(self.project_root, module_name.replace('.', os.sep) + '.py')
        if not os.path.exists(file_path):
            return None

        tree = self._get_tree(file_path)
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        return Dependency(
                            name=f"{module_name}.{variable_name}",
                            dependency_type='module_variable_definition',
                            file_path=file_path,
                            line_no=node.lineno
                        )
        return None
