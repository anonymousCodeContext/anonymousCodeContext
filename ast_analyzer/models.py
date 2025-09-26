from dataclasses import dataclass, field
from typing import Set

@dataclass(frozen=True)
class Dependency:
    """Represents a dependency found in the code."""
    name: str  # e.g., 'mrjob.hadoop.HadoopJobRunner.fs'
    dependency_type: str  # e.g., 'intra_class', 'cross_file'
    file_path: str
    line_no: int

@dataclass
class FunctionAnalysisResult:
    """Holds all dependencies for a single function."""
    function_id: str
    dependencies: Set[Dependency] = field(default_factory=set)
