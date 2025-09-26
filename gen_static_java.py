#!/usr/bin/env python3
"""
Java Static Context Generator

Generate context code for Java methods by analyzing their dependencies.
This tool provides the same functionality as gen_static.py but for Java projects.

Usage Examples:
    # Run demo with sample e-commerce project
    python gen_static_java.py --demo
    
    # Analyze specific method in a project
    python gen_static_java.py --project /path/to/java/project \
                              --file src/main/java/com/example/MyClass.java \
                              --method "src/main/java/com/example/MyClass.java::MyClass::myMethod"
    
    # Show usage help
    python gen_static_java.py --help

Features:
- Analyzes Java method dependencies across files
- Extracts source code for all dependencies
- Generates complete context excluding target method
- Supports complex inheritance and static method calls
- Works with real-world Java projects
"""

import os
import pprint
import javalang
import argparse
from typing import List, Set, Dict, Optional

from ast_analyzer_java.analyzer import JavaAstAnalyzer
from ast_analyzer_java.java_codebase_indexer import JavaCodebaseIndexer

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

class JavaClassVisitor:
    """Visitor to extract all methods from a Java file."""
    def __init__(self, relative_path: str):
        self.relative_path = relative_path
        self.methods = []
    
    def visit_compilation_unit(self, tree: javalang.tree.CompilationUnit):
        """Visit a compilation unit (Java file) to extract methods."""
        if tree.types:
            for type_decl in tree.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration):
                    self.visit_class(type_decl)
                elif isinstance(type_decl, javalang.tree.InterfaceDeclaration):
                    self.visit_interface(type_decl)
    
    def visit_class(self, class_node: javalang.tree.ClassDeclaration):
        """Visit a class declaration to extract methods."""
        class_name = class_node.name
        
        for member in class_node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                self.methods.append(f"{self.relative_path}::{class_name}::{member.name}")
            elif isinstance(member, javalang.tree.ConstructorDeclaration):
                # Constructor has the same name as the class
                self.methods.append(f"{self.relative_path}::{class_name}::{class_name}")
            # Note: We could also handle nested classes here if needed
    
    def visit_interface(self, interface_node: javalang.tree.InterfaceDeclaration):
        """Visit an interface declaration to extract methods."""
        interface_name = interface_node.name
        
        for member in interface_node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                # Interface methods (could be default or static in Java 8+)
                self.methods.append(f"{self.relative_path}::{interface_name}::{member.name}")

def get_all_methods_in_file(file_path: str, project_root: str) -> List[str]:
    """
    Parses a Java file and returns a list of all method identifiers.
    
    Args:
        file_path: The absolute path to the Java file.
        project_root: The absolute path to the project's root directory.
        
    Returns:
        A list of method identifiers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = javalang.parse.parse(content)
    except javalang.parser.JavaSyntaxError as e:
        print(f"❌ Failed to parse {file_path}: {e}")
        return []
    
    relative_path = os.path.relpath(file_path, project_root).replace(os.sep, '/')
    
    visitor = JavaClassVisitor(relative_path)
    visitor.visit_compilation_unit(tree)
    return visitor.methods

def generate_context_for_java_file(project_root: str, target_file_path: str, target_method_id: str) -> str:
    """
    Generates a context string containing the source code of all dependencies of methods
    in a Java file, excluding the target method itself.
    
    Args:
        project_root: The absolute path to the project's root directory.
        target_file_path: The absolute path to the target Java file.
        target_method_id: The identifier of the method to be excluded from analysis.
        
    Returns:
        A string containing the concatenated source code of all found dependencies.
    """
    print(f"🔬 Analyzing Java file: {target_file_path}")
    print(f"🎯 Excluding target method: {target_method_id}")
    
    # 1. Get all methods in the file
    all_methods_in_file = get_all_methods_in_file(target_file_path, project_root)
    
    # 2. Exclude the target method
    methods_to_analyze = [m for m in all_methods_in_file if m != target_method_id]
    
    print("\nMethods to analyze:")
    pprint.pprint(methods_to_analyze)
    
    # 3. Analyze dependencies for the remaining methods
    analyzer = JavaAstAnalyzer(project_root)
    all_dependencies: Set = set()
    
    for method_id in methods_to_analyze:
        try:
            analysis_result = analyzer.analyze_method(method_id)
            for dep in analysis_result.dependencies:
                all_dependencies.add(dep)  # Add the whole Dependency object
        except Exception as e:
            print(f"❌ Error analyzing {method_id}: {e}")
    
    print("\nFound dependencies:")
    pprint.pprint(all_dependencies)
    
    # 4. Index the codebase and retrieve dependency source code
    # Reuse a global indexer if available to avoid repeated indexing cost
    global _GLOBAL_JAVA_INDEXER
    try:
        _GLOBAL_JAVA_INDEXER
    except NameError:
        _GLOBAL_JAVA_INDEXER = JavaCodebaseIndexer()
    indexer = _GLOBAL_JAVA_INDEXER
    indexer.ensure_indexed(project_root)
    
    context_code = []
    processed_code_blocks = set()  # To track added code blocks and prevent duplicates
    print("\n📚 Retrieving source code for dependencies...")
    
    for dep in sorted(list(all_dependencies), key=lambda d: d.name):
        dep_name = dep.name
        code = None  # Initialize code for this dependency
        
        if dep.dependency_type == 'field' or dep.dependency_type == 'inherited_field':
            # Handle field access
            parts = dep_name.split('.')
            field_name = parts[-1]
            class_name = '.'.join(parts[:-1])
            
            # Try to find the class file
            # For simplicity, assume class is in same package structure
            class_simple_name = class_name.split('.')[-1]
            
            # Search for the class file
            for root, _, files in os.walk(project_root):
                for file in files:
                    if file == f"{class_simple_name}.java":
                        file_path = os.path.join(root, file)
                        code = indexer.get_field_code(file_path, class_simple_name, field_name)
                        if code:
                            break
                if code:
                    break
        
        elif dep.dependency_type == 'static_field':
            # Handle static field access
            parts = dep_name.split('.')
            field_name = parts[-1]
            class_name = '.'.join(parts[:-1])
            class_simple_name = class_name.split('.')[-1]
            
            # Search for the class file containing the static field
            for root, _, files in os.walk(project_root):
                for file in files:
                    if file == f"{class_simple_name}.java":
                        file_path = os.path.join(root, file)
                        code = indexer.get_static_field_code(file_path, class_simple_name, field_name)
                        if code:
                            break
                if code:
                    break
        
        elif dep.dependency_type in ('intra_class', 'cross_file', 'inherited'):
            # Handle method dependencies
            parts = dep_name.split('.')
            method_name = parts[-1]
            
            # Handle different formats of dependency names
            if dep.dependency_type == 'cross_file' and len(parts) == 2:
                # Format like: mathUtils.power or Logger.log
                object_or_class = parts[0]
                
                # Check if it's a class (starts with uppercase) or instance
                if object_or_class[0].isupper():
                    # Static method call - class name is the object
                    class_simple_name = object_or_class
                else:
                    # Instance method call - need to infer the class
                    # For now, capitalize first letter as a heuristic
                    class_simple_name = object_or_class[0].upper() + object_or_class[1:]
                
                # Search for the class file
                for root, _, files in os.walk(project_root):
                    for file in files:
                        if file == f"{class_simple_name}.java":
                            file_path = os.path.join(root, file)
                            code = indexer.get_method_code(file_path, method_name, class_simple_name)
                            if code:
                                break
                    if code:
                        break
            else:
                # Full class name format
                class_name = '.'.join(parts[:-1]) if len(parts) > 1 else None
                
                if class_name:
                    class_simple_name = class_name.split('.')[-1]
                    
                    # Search for the class file
                    for root, _, files in os.walk(project_root):
                        for file in files:
                            if file == f"{class_simple_name}.java":
                                file_path = os.path.join(root, file)
                                code = indexer.get_method_code(file_path, method_name, class_simple_name)
                                if code:
                                    break
                        if code:
                            break
                else:
                    # Standalone method (rare in Java)
                    code = None
        
        elif dep.dependency_type == 'class_instantiation':
            # Handle class instantiation - get the constructor
            class_name = dep_name
            class_simple_name = class_name.split('.')[-1]
            
            # Search for the class file
            for root, _, files in os.walk(project_root):
                for file in files:
                    if file == f"{class_simple_name}.java":
                        file_path = os.path.join(root, file)
                        # Try to get the constructor (method with same name as class)
                        constructor_code = indexer.get_method_code(file_path, class_simple_name, class_simple_name)
                        if constructor_code:
                            code = constructor_code
                        else:
                            # If no constructor found, get the whole class
                            code = indexer.get_class_code(file_path, class_simple_name)
                        if code:
                            break
                if code:
                    break
        
        # Add code to context if found and not a duplicate
        if code:
            if code not in processed_code_blocks:
                print(f"✅ Found code for: {dep_name}")
                context_code.append(f"// Source for: {dep_name}\n{code}\n")
                processed_code_blocks.add(code)
            else:
                print(f"ℹ️  Skipping duplicate code for: {dep_name}")
        else:
            print(f"❌ Could not retrieve code for: {dep_name}")
    
    return "\n".join(context_code)

def run_analysis_demo():
    """Run analysis on the demo e-commerce project."""
    # Use the demo project we created
    PROJECT_ROOT = os.path.abspath('/Users/liyichen/navigation/test_java_project')
    
    # Check if demo project exists
    if not os.path.exists(PROJECT_ROOT):
        print(f"❌ Demo project not found: {PROJECT_ROOT}")
        print("Please run the e-commerce demo setup first.")
        return
    
    # Analyze the Order.processPayment method
    TARGET_FILE = os.path.join(PROJECT_ROOT, 'src/main/java/com/example/Order.java')
    TARGET_METHOD_ID = 'src/main/java/com/example/Order.java::Order::processPayment'
    
    print("="*50)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"TARGET_FILE: {TARGET_FILE}")
    print(f"TARGET_METHOD_ID: {TARGET_METHOD_ID}")
    print("="*50)
    
    print("\n--- Running Java context generation ---")
    generated_context = generate_context_for_java_file(
        project_root=PROJECT_ROOT,
        target_file_path=TARGET_FILE,
        target_method_id=TARGET_METHOD_ID
    )
    
    print("\n--- Generated Context ---")
    if generated_context:
        print(f"Generated {len(generated_context)} characters of context")
        print("First 500 characters:")
        print(generated_context[:500])
        print("...")
    else:
        print("No context was generated.")
    print("--- End of Context ---")
    
    # Save the result to a file
    output_filename = 'java_analysis_result.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(generated_context if generated_context else "No context generated")
    print(f"\n✅ Java analysis result saved to {output_filename}")

def analyze_custom_project(project_root: str, target_file: str, target_method: str):
    """Analyze a custom Java project."""
    print(f"🔍 Analyzing Java project: {project_root}")
    print(f"📄 Target file: {target_file}")
    print(f"🎯 Target method: {target_method}")
    print("="*60)
    
    if not os.path.exists(project_root):
        print(f"❌ Project root not found: {project_root}")
        return
    
    target_file_abs = os.path.join(project_root, target_file) if not os.path.isabs(target_file) else target_file
    
    if not os.path.exists(target_file_abs):
        print(f"❌ Target file not found: {target_file_abs}")
        return
    
    print("\n--- Running Java context generation ---")
    try:
        generated_context = generate_context_for_java_file(
            project_root=project_root,
            target_file_path=target_file_abs,
            target_method_id=target_method
        )
        
        if generated_context:
            print(f"✅ Generated {len(generated_context)} characters of context")
            
            # Save to file
            output_filename = f'context_{os.path.basename(target_file).replace(".java", "")}.txt'
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(generated_context)
            print(f"💾 Context saved to: {output_filename}")
            
            # Show preview
            print("\n--- Context Preview (first 500 characters) ---")
            print(generated_context[:500])
            if len(generated_context) > 500:
                print("... [truncated]")
            print("--- End Preview ---")
        else:
            print("⚠️ No context was generated")
    
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Java Static Context Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run demo with e-commerce project
  python gen_static_java.py --demo
  
  # Analyze specific method
  python gen_static_java.py --project /path/to/java/project \\
                            --file src/main/java/com/example/MyClass.java \\
                            --method "src/main/java/com/example/MyClass.java::MyClass::myMethod"
        '''
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample e-commerce project')
    parser.add_argument('--project', type=str,
                       help='Path to Java project root directory')
    parser.add_argument('--file', type=str,
                       help='Relative path to target Java file')
    parser.add_argument('--method', type=str,
                       help='Method identifier (format: file::class::method)')
    
    args = parser.parse_args()
    
    if args.demo:
        run_analysis_demo()
    elif args.project and args.file and args.method:
        analyze_custom_project(args.project, args.file, args.method)
    else:
        parser.print_help()
        print("\n🔍 Java Static Context Generator")
        print("Choose one of the options above to analyze Java code dependencies.")

if __name__ == '__main__':
    main()