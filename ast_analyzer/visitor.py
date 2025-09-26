import ast
from typing import List, Dict, Any

class FunctionVisitor(ast.NodeVisitor):
    """
    An AST visitor that finds all dependencies within a specific function.
    It collects raw dependency "events" to be processed later.
    """
    def __init__(self, target_class_name: str = None, target_func_name: str = None):
        self.target_class_name = target_class_name
        self.target_func_name = target_func_name
        self.current_class_name = None
        self.local_variables = set()
        # A list of raw dependency events found.
        # Each event is a dictionary. e.g., {'type': 'attribute_access', 'object': 'self', 'attribute': 'fs', 'node': node}
        self.dependencies: List[Dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current_class_name = node.name
        # If a target class is specified, only visit its children.
        # Otherwise, visit all classes.
        if self.target_class_name is None or self.current_class_name == self.target_class_name:
            self.generic_visit(node)
        self.current_class_name = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # We are inside the target class, now check for the target function
        is_target_scope = (self.target_class_name is None or self.current_class_name == self.target_class_name) and \
                          (self.target_func_name is None or node.name == self.target_func_name)

        if is_target_scope:
            # Record function arguments as local variables
            if node.args:
                for arg in node.args.args:
                    self.local_variables.add(arg.arg)
                if node.args.vararg:
                    self.local_variables.add(node.args.vararg.arg)
                if node.args.kwarg:
                    self.local_variables.add(node.args.kwarg.arg)

            # This is a target scope, visit its body for dependencies
            for body_item in node.body:
                self.visit(body_item)
            
            # Clear locals after visiting the function
            self.local_variables.clear()

    def visit_Assign(self, node: ast.Assign):
        # We are inside a function. Let's see if it's an attribute assignment.
        def add_target_names(t):
            # Recursively add names from assignment targets (including tuple unpacking)
            if isinstance(t, ast.Name):
                self.local_variables.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    add_target_names(elt)

        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                event = {
                    'type': 'attribute_definition',
                    'class_name': self.current_class_name,
                    'attribute_name': target.attr,
                    'node': node
                }
                self.dependencies.append(event)
            else:
                add_target_names(target)
        
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Handle annotated assignments: x: int = 1, or self.x: int = ...
        target = node.target
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
            event = {
                'type': 'attribute_definition',
                'class_name': self.current_class_name,
                'attribute_name': target.attr,
                'node': node
            }
            self.dependencies.append(event)
        elif isinstance(target, ast.Name):
            self.local_variables.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        # for name in ... / for a, b in ...
        def add_names(t):
            if isinstance(t, ast.Name):
                self.local_variables.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    add_names(elt)
        add_names(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        # with ctx as name
        for item in node.items:
            opt = item.optional_vars
            if isinstance(opt, ast.Name):
                self.local_variables.add(opt.id)
            elif isinstance(opt, (ast.Tuple, ast.List)):
                for elt in opt.elts:
                    if isinstance(elt, ast.Name):
                        self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # except Exception as e:
        if node.name and isinstance(node.name, str):
            self.local_variables.add(node.name)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                self.local_variables.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                self.local_variables.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                self.local_variables.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                self.local_variables.add(gen.target.id)
            elif isinstance(gen.target, (ast.Tuple, ast.List)):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.local_variables.add(elt.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # e.g., self.fs or log.info
        obj = node.value
        if isinstance(obj, ast.Name):
            obj_name = obj.id
            attr_name = node.attr
            
            event = {
                'type': 'attribute_access',
                'object_name': obj_name,
                'attribute_name': attr_name,
                'node': node
            }
            self.dependencies.append(event)
        
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # This captures usages of variables.
        # We need to distinguish between local variables and module-level variables.
        if isinstance(node.ctx, ast.Load) and node.id not in self.local_variables:
            event = {
                'type': 'module_variable_access',
                'variable_name': node.id,
                'node': node
            }
            self.dependencies.append(event)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # e.g., _logs_exist(...) or self._read_logs()
        func = node.func
        
        if isinstance(func, ast.Name): # e.g., unique(...)
            event = {
                'type': 'call',
                'func_name': func.id,
                'node': node
            }
            self.dependencies.append(event)
        elif isinstance(func, ast.Attribute): # e.g., self.fs.join(...)
            # Capture nested attribute chains like self.fs.hadoop.get_hadoop_bin
            chain: List[str] = []
            value = func
            while isinstance(value, ast.Attribute):
                chain.insert(0, value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                obj_name = value.id
                event = {
                    'type': 'method_call',
                    'object_name': obj_name,
                    'method_name': chain[-1] if chain else func.attr,
                    'full_attr_chain': [obj_name] + chain,
                    'node': node
                }
                self.dependencies.append(event)

        self.generic_visit(node)
