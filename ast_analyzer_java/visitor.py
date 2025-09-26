import javalang
from typing import List, Dict, Any, Set, Optional

class JavaMethodVisitor:
    """
    A visitor that traverses a Java AST to find dependencies within a specific method.
    """
    def __init__(self, target_class_name: str = None, target_method_name: str = None):
        self.target_class_name = target_class_name
        self.target_method_name = target_method_name
        self.current_class_name = None
        self.local_variables: Set[str] = set()
        self.class_fields: Set[str] = set()
        self.dependencies: List[Dict[str, Any]] = []
        self.imports: Dict[str, str] = {}  # simple_name -> fully_qualified_name
        
    def visit_tree(self, tree: javalang.tree.CompilationUnit):
        """Visit the entire compilation unit (Java file)."""
        # First, collect all imports
        if tree.imports:
            for import_decl in tree.imports:
                if import_decl.static:
                    continue  # Skip static imports for now
                path = import_decl.path
                if import_decl.wildcard:
                    # For wildcard imports, we can't map specific names
                    # We'll handle this in resolution logic
                    pass
                else:
                    # Regular import: com.example.MyClass -> MyClass maps to com.example.MyClass
                    simple_name = path.split('.')[-1]
                    self.imports[simple_name] = path
        
        # Then visit all type declarations (classes, interfaces, enums)
        if tree.types:
            for type_decl in tree.types:
                self.visit_type_declaration(type_decl)
    
    def visit_type_declaration(self, node):
        """Visit a type declaration (class, interface, or enum)."""
        if isinstance(node, javalang.tree.ClassDeclaration):
            self.current_class_name = node.name
            
            # Collect class fields
            for member in node.body:
                if isinstance(member, javalang.tree.FieldDeclaration):
                    for declarator in member.declarators:
                        self.class_fields.add(declarator.name)
            
            # If we're looking for a specific class, check if this is it
            if self.target_class_name is None or self.current_class_name == self.target_class_name:
                # Visit all methods in the class
                for member in node.body:
                    if isinstance(member, javalang.tree.MethodDeclaration):
                        self.visit_method(member)
                    elif isinstance(member, javalang.tree.ConstructorDeclaration):
                        # Treat constructors as methods with the class name
                        self.visit_constructor(member)
            
            self.current_class_name = None
            self.class_fields.clear()
    
    def visit_method(self, method: javalang.tree.MethodDeclaration):
        """Visit a method declaration."""
        # Check if this is the target method
        if self.target_method_name and method.name != self.target_method_name:
            return
        
        # Clear local variables from previous methods
        self.local_variables.clear()
        
        # Add method parameters as local variables
        if method.parameters:
            for param in method.parameters:
                self.local_variables.add(param.name)
        
        # Visit the method body
        if method.body:
            for statement in method.body:
                self.visit_statement(statement)
    
    def visit_constructor(self, constructor: javalang.tree.ConstructorDeclaration):
        """Visit a constructor declaration."""
        # If we're looking for a constructor, the method name should match the class name
        if self.target_method_name and self.target_method_name != self.current_class_name:
            return
        
        # Clear local variables
        self.local_variables.clear()
        
        # Add constructor parameters as local variables
        if constructor.parameters:
            for param in constructor.parameters:
                self.local_variables.add(param.name)
        
        # Visit the constructor body
        if constructor.body:
            for statement in constructor.body:
                self.visit_statement(statement)
    
    def visit_statement(self, stmt):
        """Visit a statement recursively."""
        if stmt is None:
            return
            
        # Handle different types of statements
        if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
            # Add local variables
            for declarator in stmt.declarators:
                self.local_variables.add(declarator.name)
                if declarator.initializer:
                    self.visit_expression(declarator.initializer)
        
        elif isinstance(stmt, javalang.tree.StatementExpression):
            self.visit_expression(stmt.expression)
        
        elif isinstance(stmt, javalang.tree.IfStatement):
            self.visit_expression(stmt.condition)
            self.visit_statement(stmt.then_statement)
            if stmt.else_statement:
                self.visit_statement(stmt.else_statement)
        
        elif isinstance(stmt, javalang.tree.WhileStatement):
            self.visit_expression(stmt.condition)
            self.visit_statement(stmt.body)
        
        elif isinstance(stmt, javalang.tree.ForStatement):
            # Traditional for loop
            if hasattr(stmt, 'control') and stmt.control:
                control = stmt.control
                if isinstance(control, javalang.tree.ForControl):
                    # Traditional for loop: for(init; condition; update)
                    if hasattr(control, 'init') and control.init:
                        self.visit_statement(control.init)
                    if hasattr(control, 'condition') and control.condition:
                        self.visit_expression(control.condition)
                    if hasattr(control, 'update') and control.update:
                        for expr in control.update:
                            self.visit_expression(expr)
                elif isinstance(control, javalang.tree.EnhancedForControl):
                    # Enhanced for loop: for(Type var : iterable)
                    if hasattr(control.var, 'name'):
                        self.local_variables.add(control.var.name)
                    elif hasattr(control.var, 'declarators'):
                        # VariableDeclaration with declarators
                        for declarator in control.var.declarators:
                            self.local_variables.add(declarator.name)
                    self.visit_expression(control.iterable)
            
            if stmt.body:
                self.visit_statement(stmt.body)
        
        elif isinstance(stmt, javalang.tree.EnhancedForControl):
            if hasattr(stmt.var, 'name'):
                self.local_variables.add(stmt.var.name)
            elif hasattr(stmt.var, 'declarators'):
                for declarator in stmt.var.declarators:
                    self.local_variables.add(declarator.name)
            self.visit_expression(stmt.iterable)
        
        elif isinstance(stmt, javalang.tree.BlockStatement):
            for s in stmt.statements:
                self.visit_statement(s)
        
        elif isinstance(stmt, javalang.tree.ReturnStatement):
            if stmt.expression:
                self.visit_expression(stmt.expression)
        
        elif isinstance(stmt, javalang.tree.ThrowStatement):
            self.visit_expression(stmt.expression)
        
        elif isinstance(stmt, javalang.tree.TryStatement):
            for s in stmt.block:
                self.visit_statement(s)
            for catch_clause in stmt.catches or []:
                # Add exception variable as local
                self.local_variables.add(catch_clause.parameter.name)
                for s in catch_clause.block:
                    self.visit_statement(s)
            if stmt.finally_block:
                for s in stmt.finally_block:
                    self.visit_statement(s)
        
        # Handle other statement types as needed
        elif hasattr(stmt, '__dict__'):
            # Generic traversal for unhandled statement types
            for attr_value in stmt.__dict__.values():
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if hasattr(item, '__dict__'):
                            self.visit_statement(item)
                elif hasattr(attr_value, '__dict__'):
                    self.visit_statement(attr_value)
    
    def visit_expression(self, expr):
        """Visit an expression recursively."""
        if expr is None:
            return
        
        # Handle method invocations
        if isinstance(expr, javalang.tree.MethodInvocation):
            method_name = expr.member
            qualifier = expr.qualifier
            
            if qualifier:
                # Qualified method call (e.g., object.method() or Class.staticMethod())
                if isinstance(qualifier, str):
                    # Could be 'this', 'super', or a variable/class name
                    if qualifier == 'this':
                        # Intra-class method call
                        event = {
                            'type': 'method_call',
                            'object_name': 'this',
                            'method_name': method_name,
                            'line_no': 0  # Line number would need to be tracked
                        }
                        self.dependencies.append(event)
                    elif qualifier == 'super':
                        # Parent class method call
                        event = {
                            'type': 'super_method_call',
                            'method_name': method_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
                    elif qualifier[0].isupper():
                        # Likely a class name (static method call)
                        event = {
                            'type': 'static_method_call',
                            'class_name': qualifier,
                            'method_name': method_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
                    else:
                        # Variable method call
                        event = {
                            'type': 'method_call',
                            'object_name': qualifier,
                            'method_name': method_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
                elif isinstance(qualifier, javalang.tree.MemberReference):
                    # Chained method call (e.g., obj.field.method())
                    chain = self._build_member_chain(qualifier)
                    chain.append(method_name)
                    event = {
                        'type': 'chained_method_call',
                        'chain': chain,
                        'line_no': 0
                    }
                    self.dependencies.append(event)
            else:
                # Unqualified method call (could be local method or imported static method)
                event = {
                    'type': 'call',
                    'func_name': method_name,
                    'line_no': 0
                }
                self.dependencies.append(event)
            
            # Visit method arguments
            if expr.arguments:
                for arg in expr.arguments:
                    self.visit_expression(arg)
        
        # Handle member references (field access)
        elif isinstance(expr, javalang.tree.MemberReference):
            member_name = expr.member
            qualifier = expr.qualifier
            
            if qualifier:
                if isinstance(qualifier, str):
                    if qualifier == 'this':
                        # Accessing instance field
                        event = {
                            'type': 'field_access',
                            'object_name': 'this',
                            'field_name': member_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
                    elif qualifier[0].isupper():
                        # Likely accessing static field
                        event = {
                            'type': 'static_field_access',
                            'class_name': qualifier,
                            'field_name': member_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
                    else:
                        # Accessing field of an object
                        event = {
                            'type': 'field_access',
                            'object_name': qualifier,
                            'field_name': member_name,
                            'line_no': 0
                        }
                        self.dependencies.append(event)
        
        # Handle class instantiation
        elif isinstance(expr, javalang.tree.ClassCreator):
            class_type = expr.type.name
            event = {
                'type': 'class_instantiation',
                'class_name': class_type,
                'line_no': 0
            }
            self.dependencies.append(event)
            
            # Visit constructor arguments
            if expr.arguments:
                for arg in expr.arguments:
                    self.visit_expression(arg)
        
        # Handle variable references
        elif isinstance(expr, javalang.tree.MemberReference) and expr.qualifier == '':
            var_name = expr.member
            if var_name not in self.local_variables and var_name not in self.class_fields:
                # Could be a class name or external reference
                event = {
                    'type': 'variable_access',
                    'variable_name': var_name,
                    'line_no': 0
                }
                self.dependencies.append(event)
        
        # Handle binary operations
        elif isinstance(expr, javalang.tree.BinaryOperation):
            self.visit_expression(expr.operandl)
            self.visit_expression(expr.operandr)
        
        # Handle unary operations
        elif hasattr(expr, 'expression'):
            self.visit_expression(expr.expression)
        
        # Handle other expression types recursively
        elif hasattr(expr, '__dict__'):
            for attr_value in expr.__dict__.values():
                if isinstance(attr_value, list):
                    for item in attr_value:
                        if hasattr(item, '__dict__'):
                            self.visit_expression(item)
                elif hasattr(attr_value, '__dict__'):
                    self.visit_expression(attr_value)
    
    def _build_member_chain(self, member_ref) -> List[str]:
        """Build a chain of member accesses."""
        chain = []
        current = member_ref
        while isinstance(current, javalang.tree.MemberReference):
            chain.insert(0, current.member)
            current = current.qualifier
        if isinstance(current, str):
            chain.insert(0, current)
        return chain
