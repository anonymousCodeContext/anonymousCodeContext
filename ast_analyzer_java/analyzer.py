import os
import javalang
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field

from .visitor import JavaMethodVisitor
from .models import JavaFunctionAnalysisResult, JavaDependency, JavaClassInfo

class JavaAstAnalyzer:
    """
    Analyzer for Java code that extracts dependencies from methods.
    """
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self._tree_cache: Dict[str, javalang.tree.CompilationUnit] = {}
        self._class_info_cache: Dict[str, JavaClassInfo] = {}
        self._package_map: Dict[str, str] = {}  # class_name -> package_name
        
    def _get_tree(self, file_path: str) -> javalang.tree.CompilationUnit:
        """Parse and cache a Java file's AST."""
        if file_path in self._tree_cache:
            return self._tree_cache[file_path]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            tree = javalang.parse.parse(code)
            self._tree_cache[file_path] = tree
            
            # Extract package name if present
            if tree.package:
                package_name = tree.package.name
                # Map all classes in this file to this package
                if tree.types:
                    for type_decl in tree.types:
                        if hasattr(type_decl, 'name'):
                            self._package_map[type_decl.name] = package_name
            
            return tree
        except javalang.parser.JavaSyntaxError as e:
            print(f"❌ Failed to parse {file_path}: {e}")
            return None
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get the relative path from project root."""
        return os.path.relpath(file_path, self.project_root).replace(os.sep, '/')
    
    def _get_class_info(self, class_name: str, file_path: str) -> Optional[JavaClassInfo]:
        """Get information about a class including its methods and fields."""
        fq_class_name = self._get_fully_qualified_name(class_name, file_path)
        
        if fq_class_name in self._class_info_cache:
            return self._class_info_cache[fq_class_name]
        
        tree = self._get_tree(file_path)
        if not tree:
            return None
        
        # Find the class in the tree
        class_node = None
        if tree.types:
            for type_decl in tree.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration) and type_decl.name == class_name:
                    class_node = type_decl
                    break
        
        if not class_node:
            return None
        
        # Collect class information
        methods = set()
        fields = set()
        
        for member in class_node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                methods.add(member.name)
            elif isinstance(member, javalang.tree.ConstructorDeclaration):
                methods.add(class_name)  # Constructor has same name as class
            elif isinstance(member, javalang.tree.FieldDeclaration):
                for declarator in member.declarators:
                    fields.add(declarator.name)
        
        # Get parent class and interfaces
        parent_class = None
        interfaces = set()
        
        if class_node.extends:
            parent_class = class_node.extends.name
        
        if class_node.implements:
            for interface in class_node.implements:
                interfaces.add(interface.name)
        
        class_info = JavaClassInfo(
            name=fq_class_name,
            file_path=file_path,
            methods=methods,
            fields=fields,
            parent_class=parent_class,
            interfaces=interfaces
        )
        
        self._class_info_cache[fq_class_name] = class_info
        return class_info
    
    def _get_fully_qualified_name(self, class_name: str, file_path: str) -> str:
        """Get the fully qualified name of a class."""
        # Try to get from cache first
        if class_name in self._package_map:
            return f"{self._package_map[class_name]}.{class_name}"
        
        # Parse the file to get package info
        tree = self._get_tree(file_path)
        if tree and tree.package:
            return f"{tree.package.name}.{class_name}"
        
        # No package, just return class name
        return class_name
    
    def _resolve_import(self, simple_name: str, imports: Dict[str, str]) -> Optional[str]:
        """Resolve a simple class name to its fully qualified name using imports."""
        if simple_name in imports:
            return imports[simple_name]
        return None
    
    def _analyze_method(self, file_path: str, method_name: str, class_name: Optional[str] = None) -> JavaFunctionAnalysisResult:
        """Analyze a specific method to find its dependencies."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        tree = self._get_tree(file_path)
        if not tree:
            raise ValueError(f"Could not parse file: {file_path}")
        
        # Get class info if analyzing a method in a class
        class_info = None
        if class_name:
            class_info = self._get_class_info(class_name, file_path)
            if not class_info:
                raise ValueError(f"Could not find class {class_name} in {file_path}")
        
        # Create visitor and visit the tree
        visitor = JavaMethodVisitor(target_class_name=class_name, target_method_name=method_name)
        visitor.visit_tree(tree)
        
        # Build method ID
        relative_path = self._get_relative_path(file_path)
        if class_name:
            method_id = f"{relative_path}::{class_name}::{method_name}"
        else:
            method_id = f"{relative_path}::{method_name}"
        
        result = JavaFunctionAnalysisResult(method_id=method_id)
        
        # Get package name for current file
        current_package = tree.package.name if tree.package else ""
        
        # Process dependencies found by visitor
        for event in visitor.dependencies:
            dep_type = "unknown"
            dep_name = "unknown"
            
            if event['type'] == 'method_call':
                obj_name = event['object_name']
                method = event['method_name']
                
                if obj_name == 'this' and class_info:
                    # Check if method is in current class
                    if method in class_info.methods:
                        dep_type = 'intra_class'
                        dep_name = f"{class_info.name}.{method}"
                    elif class_info.parent_class:
                        # Could be inherited method
                        dep_type = 'inherited'
                        dep_name = f"{class_info.parent_class}.{method}"
                else:
                    # External method call
                    dep_type = 'cross_file'
                    # Try to resolve the class of the object
                    if obj_name in visitor.imports:
                        dep_name = f"{visitor.imports[obj_name]}.{method}"
                    else:
                        dep_name = f"{obj_name}.{method}"
            
            elif event['type'] == 'super_method_call':
                method = event['method_name']
                if class_info and class_info.parent_class:
                    dep_type = 'inherited'
                    dep_name = f"{class_info.parent_class}.{method}"
            
            elif event['type'] == 'static_method_call':
                class_called = event['class_name']
                method = event['method_name']
                dep_type = 'cross_file'
                
                # Try to resolve full class name
                if class_called in visitor.imports:
                    dep_name = f"{visitor.imports[class_called]}.{method}"
                else:
                    dep_name = f"{class_called}.{method}"
            
            elif event['type'] == 'call':
                func_name = event['func_name']
                # Could be a method in the same class or a static import
                if class_info and func_name in class_info.methods:
                    dep_type = 'intra_class'
                    dep_name = f"{class_info.name}.{func_name}"
                else:
                    dep_type = 'cross_file'
                    dep_name = func_name
            
            elif event['type'] == 'field_access':
                obj_name = event['object_name']
                field = event['field_name']
                
                if obj_name == 'this' and class_info:
                    if field in class_info.fields:
                        dep_type = 'field'
                        dep_name = f"{class_info.name}.{field}"
                    elif class_info.parent_class:
                        dep_type = 'inherited_field'
                        dep_name = f"{class_info.parent_class}.{field}"
                else:
                    dep_type = 'external_field'
                    dep_name = f"{obj_name}.{field}"
            
            elif event['type'] == 'static_field_access':
                class_name_ref = event['class_name']
                field = event['field_name']
                dep_type = 'static_field'
                
                if class_name_ref in visitor.imports:
                    dep_name = f"{visitor.imports[class_name_ref]}.{field}"
                else:
                    dep_name = f"{class_name_ref}.{field}"
            
            elif event['type'] == 'class_instantiation':
                class_instantiated = event['class_name']
                dep_type = 'class_instantiation'
                
                if class_instantiated in visitor.imports:
                    dep_name = visitor.imports[class_instantiated]
                else:
                    dep_name = class_instantiated
            
            elif event['type'] == 'chained_method_call':
                chain = event['chain']
                dep_type = 'chained_call'
                dep_name = '.'.join(chain)
            
            # Add dependency if resolved
            if dep_type != "unknown":
                result.dependencies.add(
                    JavaDependency(
                        name=dep_name,
                        dependency_type=dep_type,
                        file_path=file_path,
                        line_no=event.get('line_no', 0)
                    )
                )
        
        return result
    
    def analyze_method(self, method_id: str) -> JavaFunctionAnalysisResult:
        """
        Analyze a method given its ID.
        
        Args:
            method_id: Method identifier in format:
                      'path/to/File.java::ClassName::methodName' or
                      'path/to/File.java::methodName' for static/top-level methods
        
        Returns:
            JavaFunctionAnalysisResult containing all dependencies
        """
        parts = method_id.split('::')
        
        if len(parts) == 3:
            file_path_str, class_name, method_name = parts
        elif len(parts) == 2:
            file_path_str, method_name = parts
            class_name = None
        else:
            raise ValueError(f"Invalid method_id format: {method_id}")
        
        file_path = os.path.join(self.project_root, file_path_str)
        
        return self._analyze_method(file_path, method_name, class_name)
    
    def find_field_definition(self, class_name: str, field_name: str) -> Optional[JavaDependency]:
        """
        Find where a field is defined in a class.
        
        Args:
            class_name: Fully qualified class name
            field_name: Name of the field
            
        Returns:
            JavaDependency pointing to the field definition
        """
        # Search through cached trees for the class
        for file_path, tree in self._tree_cache.items():
            if tree and tree.types:
                for type_decl in tree.types:
                    if isinstance(type_decl, javalang.tree.ClassDeclaration):
                        # Check if this is the class we're looking for
                        fq_name = self._get_fully_qualified_name(type_decl.name, file_path)
                        if fq_name == class_name or type_decl.name == class_name:
                            # Search for the field
                            for member in type_decl.body:
                                if isinstance(member, javalang.tree.FieldDeclaration):
                                    for declarator in member.declarators:
                                        if declarator.name == field_name:
                                            return JavaDependency(
                                                name=f"{class_name}.{field_name}",
                                                dependency_type='field_definition',
                                                file_path=file_path,
                                                line_no=0  # Would need line tracking
                                            )
        return None



