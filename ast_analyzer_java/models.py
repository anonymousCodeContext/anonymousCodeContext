from dataclasses import dataclass, field
from typing import Set

@dataclass(frozen=True)
class JavaDependency:
    """Represents a dependency found in Java code."""
    name: str  # e.g., 'com.example.MyClass.myMethod'
    dependency_type: str  # e.g., 'intra_class', 'cross_file', 'external_library'
    file_path: str
    line_no: int

@dataclass
class JavaFunctionAnalysisResult:
    """Holds all dependencies for a single Java method."""
    method_id: str  # e.g., 'com/example/MyClass.java::MyClass::myMethod'
    dependencies: Set[JavaDependency] = field(default_factory=set)

@dataclass
class JavaClassInfo:
    """Holds information about a Java class."""
    name: str  # Fully qualified class name
    file_path: str
    methods: Set[str] = field(default_factory=set)
    fields: Set[str] = field(default_factory=set)
    parent_class: str = None  # Fully qualified parent class name
    interfaces: Set[str] = field(default_factory=set)  # Fully qualified interface names
    inherited_methods: Set[str] = field(default_factory=set)
    inherited_fields: Set[str] = field(default_factory=set)
