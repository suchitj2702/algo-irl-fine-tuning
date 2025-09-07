from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class LLMTaskConfig:
    """
    Generic LLM task configuration that works with any provider.
    This replaces the provider-specific LLMConfig.
    """
    task_type: str  # "problemGeneration", "companyDataGeneration", "customPromptTransform"
    provider: str   # "claude", "gpt", "gemini"
    model: Optional[str] = None  # Will use provider default if not specified
    api_key: Optional[str] = None  # Will use environment variable if not specified
    max_output_tokens: int = 15000
    thinking_enabled: bool = False
    thinking_budget: int = 0
    custom_params: Optional[Dict[str, Any]] = None


class ProblemDifficulty(Enum):
    """Problem difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


@dataclass
class TestCase:
    """Represents a test case for a problem"""
    stdin: str
    expectedStdout: str
    is_sample: bool = True
    explanation: Optional[str] = None

    def __post_init__(self):
        # Allow initialization with 'input' and 'output' for backward compatibility
        # but prioritize 'stdin' and 'expectedStdout'
        if hasattr(self, 'input') and not hasattr(self, 'stdin'):
            self.stdin = self.input
        if hasattr(self, 'output') and not hasattr(self, 'expectedStdout'):
            self.expectedStdout = self.output



@dataclass
class Problem:
    """Represents a coding problem"""
    id: str
    title: str
    difficulty: ProblemDifficulty
    description: str
    constraints: List[str]
    categories: List[str]
    test_cases: List[TestCase]
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None
    default_user_code: Optional[str] = None


@dataclass
class Company:
    """Represents a company profile with extended context"""
    # Core fields (backward compatible)
    id: str
    name: str
    domain: str
    products: List[str]
    technologies: List[str]
    interview_focus: List[str]
    
    # Optional fields for enhanced context
    description: Optional[str] = None
    logo_url: Optional[str] = None
    engineering_challenges: Optional[Dict[str, List[str]]] = None
    scale_metrics: Optional[Dict[str, str]] = None
    tech_stack_layers: Optional[Dict[str, List[str]]] = None
    problem_domains: Optional[List[str]] = None
    industry_buzzwords: Optional[List[str]] = None
    notable_systems: Optional[List[str]] = None
    data_processing_patterns: Optional[List[str]] = None
    optimization_priorities: Optional[List[str]] = None
    analogy_patterns: Optional[Dict[str, List[Dict[str, str]]]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """Create Company from dictionary, handling both old and new formats"""
        # Handle the old 'interviewFocus' key name
        if 'interviewFocus' in data and 'interview_focus' not in data:
            data['interview_focus'] = data.pop('interviewFocus')
        
        # Handle 'logoUrl' -> 'logo_url'
        if 'logoUrl' in data and 'logo_url' not in data:
            data['logo_url'] = data.pop('logoUrl')
            
        # Extract only the fields that exist in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)


@dataclass
class ExtractedProblemInfo:
    """Extracted key information from a problem"""
    title: str
    difficulty: ProblemDifficulty
    description: str
    constraints: List[str]
    categories: List[str]
    time_complexity: Optional[str]
    space_complexity: Optional[str]
    keywords: List[str]
    core_algorithms: List[str]
    data_structures: List[str]
    default_user_code: Optional[str] = None
    test_cases: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExtractedCompanyInfo:
    """Extracted key information from a company"""
    name: str
    domain: str
    products: List[str]
    technologies: List[str]
    interview_focus: List[str]
    relevant_keywords: List[str]


@dataclass
class TransformationContext:
    """Enhanced prompt context combining problem and company information"""
    problem: ExtractedProblemInfo
    company: ExtractedCompanyInfo
    relevance_score: float
    suggested_analogy_points: List[str]


@dataclass
class StructuredScenario:
    """Structured sections of a transformed problem scenario"""
    title: str
    background: str
    problem_statement: str
    function_signature: str
    constraints: List[str]
    examples: List[Dict[str, Any]]
    requirements: List[str]
    function_mapping: Dict[str, str]


class RoleFamily(Enum):
    """Software engineering role families for targeted transformations"""
    BACKEND_SYSTEMS = "backend"
    ML_DATA = "ml"
    FRONTEND_FULLSTACK = "frontend" 
    INFRASTRUCTURE_PLATFORM = "infrastructure"
    SECURITY_RELIABILITY = "security"
