#!/usr/bin/env python3
"""
Algo-IRL Problem Transformation Script
Transforms coding problems into company and role-specific interview scenarios

Requirements:
    pip install openai               # For OpenAI GPT models
    pip install anthropic            # For Anthropic Claude models  
    pip install google-generativeai  # For Google Gemini models
    pip install python-dotenv        # For loading environment variables
    pip install firebase-admin       # Optional: For Firestore integration

Environment Variables:
    OPENAI_API_KEY      - Your OpenAI API key (for --provider gpt)
    ANTHROPIC_API_KEY   - Your Anthropic API key (for --provider claude) 
    GOOGLE_API_KEY      - Your Google API key (for --provider gemini)

Usage:
    python -m transform_generation.problem_transformer <company_name> <problem_id> [options]

Options:
    --provider <name>    LLM provider: claude (default), gpt, or gemini
    --role <type>        Engineering role: backend, ml, frontend, infrastructure, or security
    --use-firestore      Use Firestore for company data instead of local cache

Examples:
    # Basic transformation (uses local cache for company data)
    python -m transform_generation.problem_transformer Google two-sum
    
    # With specific provider
    python -m transform_generation.problem_transformer Google two-sum --provider gpt
    
    # With role specification (creates role-specific scenario)
    python -m transform_generation.problem_transformer Google two-sum --role backend
    python -m transform_generation.problem_transformer Meta graph-traversal --role ml
    python -m transform_generation.problem_transformer Amazon two-sum --role infrastructure
    
    # Combined: specific provider and role
    python -m transform_generation.problem_transformer Stripe payment-processing --provider gemini --role security
    
    # Use Firestore instead of local cache (requires Firebase setup)
    python -m transform_generation.problem_transformer Google two-sum --use-firestore

Notes:
    - Company data is cached locally in company_data_cache/ directory
    - First run for a company will generate and cache its data
    - Role options create specialized scenarios for different engineering positions
    - Transformations are saved in transformations/<provider>/ directory

"""

import re
import json
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import asdict, fields
from pathlib import Path
from .models import (
    LLMTaskConfig,
    ProblemDifficulty,
    TestCase,
    Problem,
    Company,
    ExtractedProblemInfo,
    ExtractedCompanyInfo,
    TransformationContext,
    StructuredScenario,
    RoleFamily,
)
from .providers import get_provider, TaskType, list_available_providers
import logging
from .scenario_parser import parse_scenario_sections


# Firebase Admin SDK
try:
    from common.firestore_utils import db, FIREBASE_AVAILABLE
except ImportError:
    db = None
    FIREBASE_AVAILABLE = False
    logging.getLogger(__name__).warning("Could not import Firestore utils. Firestore functionality will be unavailable.")

# Load environment variables from .env.local file
try:
    from dotenv import load_dotenv
    load_dotenv('.env.local')
except ImportError:
    print("dotenv not found. Please install with 'pip install python-dotenv'. Assuming environment variables are set.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: LLM libraries are now imported within provider modules
# No need to import them here





class CompanyCache:
    """Simple local cache for company data"""
    
    def __init__(self, cache_dir: str = "company_data_cache", provider_name: str = "claude"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "companies.json"
        self.companies: Dict[str, Company] = {}
        self.provider_name = provider_name
        self._load_cache()
    
    def _load_cache(self):
        """Load cached companies from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for name, company_data in data.items():
                        self.companies[name.lower()] = Company.from_dict(company_data)
                logger.info(f"Loaded {len(self.companies)} companies from cache")
            except Exception as e:
                logger.warning(f"Could not load company cache: {e}")
                self.companies = {}
    
    def _save_cache(self):
        """Save companies to cache file"""
        try:
            data = {}
            for name, company in self.companies.items():
                data[company.name] = asdict(company)
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save company cache: {e}")
    
    def get_company(self, name: str) -> Optional[Company]:
        """Get company from cache or generate if not exists"""
        normalized_name = name.lower()
        
        if normalized_name in self.companies:
            return self.companies[normalized_name]
        
        # Generate new company data using dedicated provider for company generation
        company = self._generate_company_data(name)
        if company:
            self.companies[normalized_name] = company
            self._save_cache()
        return company
    
    def _generate_company_data(self, name: str) -> Optional[Company]:
        """Generate comprehensive company data using LLM with COMPANY_DATA_GENERATION task type"""
        try:
            # Create a dedicated provider for company data generation
            # This uses the optimized model configuration for this specific task:
            # - Claude: claude-3-5-haiku-20241022 (faster, cheaper model)
            # - GPT: gpt-3.5-turbo (cost-effective for structured data)
            # - Gemini: gemini-1.5-flash-latest (fast generation)
            provider = get_provider(
                provider_name=self.provider_name,
                task_type=TaskType.COMPANY_DATA_GENERATION
            )
            
            # Log the provider configuration being used
            config_summary = provider.get_config_summary()
            logger.info(f"Generating company data for '{name}' using {config_summary['provider']} ({config_summary['model']})")
            
            prompt = f'''
I need detailed information about the company "{name}" in JSON format.

REQUIRED FIELDS (for backward compatibility):
Please provide the following information:
1. A concise description of what the company does
2. The company's primary domain/industry
3. List of main products or services (at least 3-5)
4. List of key technologies used at the company (at least 3-5)
5. List of typical interview focus areas for engineers at this company (at least 3-5)
6. A URL to the company's logo (if well-known)

ENHANCED FIELDS (for richer context):
7. Engineering challenges organized by category
8. Scale metrics (e.g., number of users, requests per day)
9. Tech stack organized by layer (frontend, backend, data, infrastructure)
10. Problem domains the company focuses on
11. Industry-specific buzzwords and terminology
12. Notable internal systems or technologies
13. Data processing patterns used
14. Optimization priorities
15. Company-specific analogy patterns for different data structures

Format your response as valid JSON ONLY with these exact fields:
{{
  "description": "string description here",
  "domain": "string domain here",
  "products": ["product1", "product2", "product3"],
  "technologies": ["tech1", "tech2", "tech3"],
  "interview_focus": ["focus1", "focus2", "focus3"],
  "logo_url": "https://example.com/logo.png or null if unknown",
  
  "engineering_challenges": {{
    "scalability": ["challenge1", "challenge2"],
    "reliability": ["challenge3"],
    "performance": ["challenge4"]
  }},
  "scale_metrics": {{
    "users": "2B+ monthly active",
    "requests_per_day": "100B+"
  }},
  "tech_stack_layers": {{
    "frontend": ["React", "TypeScript"],
    "backend": ["Java", "Python"]
  }},
  "problem_domains": ["distributed_systems", "machine_learning"],
  "industry_buzzwords": ["term1", "term2"],
  "notable_systems": ["System1", "System2"],
  "data_processing_patterns": ["streaming", "batch"],
  "optimization_priorities": ["latency", "throughput"],
  
  "analogy_patterns": {{
    "Array": [
      {{"context": "search", "analogy": "{name} search results ranking"}}
    ],
    "Graph": [
      {{"context": "social", "analogy": "{name} user network"}}
    ]
  }}
}}

Do not include any text before or after the JSON.
'''
            
            system_prompt = "You are an expert on technology companies and their engineering practices."
            response = provider.generate_completion(system_prompt, prompt)
            
            # Parse JSON response
            company_data = json.loads(response)
            
            # Add ID and name if not present
            if 'id' not in company_data:
                company_data['id'] = name.lower().replace(' ', '-')
            if 'name' not in company_data:
                company_data['name'] = name
            
            return Company.from_dict(company_data)
            
        except Exception as e:
            logger.error(f"Failed to generate company data for '{name}': {e}")
            # Return basic company data as fallback
            return Company(
                id=name.lower().replace(' ', '-'),
                name=name,
                domain='technology',
                products=['Product 1', 'Product 2', 'Product 3'],
                technologies=['Technology 1', 'Technology 2', 'Technology 3'],
                interview_focus=['algorithms', 'system design', 'problem solving']
            )


class ProblemTransformer:
    """Main class for transforming problems into company-specific scenarios"""

    # Common data structures and algorithms
    DATA_STRUCTURES = [
        'Array', 'LinkedList', 'Stack', 'Queue', 'HashMap', 'HashSet', 'HashTable',
        'Tree', 'BinaryTree', 'BinarySearchTree', 'Heap', 'PriorityQueue',
        'Graph', 'Trie', 'Matrix', 'String'
    ]

    ALGORITHMS = [
        'BFS', 'DFS', 'BinarySearch', 'Sorting', 'MergeSort', 'QuickSort',
        'DynamicProgramming', 'Recursion', 'Backtracking', 'Greedy', 'TwoPointers',
        'SlidingWindow', 'Hashing', 'Memoization', 'UnionFind', 'TopologicalSort'
    ]

    # Role-specific configurations
    ROLE_CONFIGURATIONS = {
        RoleFamily.BACKEND_SYSTEMS: {
            'title': 'Backend & Distributed Systems Engineer',
            'focus_areas': [
                'API design and optimization',
                'Distributed systems architecture',
                'Database performance',
                'Microservices orchestration',
                'Message queue systems'
            ],
            'relevant_problems': [
                'Rate limiting and throttling',
                'Caching strategies',
                'Data consistency',
                'Service discovery',
                'Load balancing'
            ],
            'key_metrics': ['requests per second', 'p99 latency', 'error rate', 'throughput'],
            'typical_scenarios': [
                'Design a distributed rate limiter',
                'Optimize database query performance',
                'Implement distributed caching',
                'Build event-driven architectures'
            ],
            'algorithm_preferences': ['hashing', 'consistent hashing', 'bloom filters', 'consensus algorithms']
        },
        
        RoleFamily.ML_DATA: {
            'title': 'Machine Learning & Data Engineer',
            'focus_areas': [
                'Data pipeline design',
                'Feature engineering',
                'Model serving infrastructure',
                'Stream processing',
                'Data warehousing'
            ],
            'relevant_problems': [
                'Real-time feature computation',
                'Data deduplication',
                'Streaming aggregations',
                'Batch processing optimization',
                'Data validation'
            ],
            'key_metrics': ['data freshness', 'pipeline latency', 'processing throughput', 'model accuracy'],
            'typical_scenarios': [
                'Build real-time ML feature pipelines',
                'Design streaming ETL systems',
                'Implement A/B testing frameworks',
                'Optimize data sampling strategies'
            ],
            'algorithm_preferences': ['sampling algorithms', 'streaming algorithms', 'probabilistic data structures', 'MapReduce patterns']
        },
        
        RoleFamily.FRONTEND_FULLSTACK: {
            'title': 'Frontend & Full-Stack Engineer',
            'focus_areas': [
                'UI performance optimization',
                'State management',
                'Real-time collaboration',
                'Progressive web apps',
                'API integration'
            ],
            'relevant_problems': [
                'Virtual scrolling',
                'Debouncing and throttling',
                'Offline synchronization',
                'Optimistic UI updates',
                'Client-side caching'
            ],
            'key_metrics': ['first contentful paint', 'time to interactive', 'bundle size', 'frame rate'],
            'typical_scenarios': [
                'Implement infinite scrolling',
                'Build real-time collaborative editing',
                'Design offline-first applications',
                'Optimize rendering performance'
            ],
            'algorithm_preferences': ['tree diffing', 'memoization', 'virtual DOM algorithms', 'event delegation']
        },
        
        RoleFamily.INFRASTRUCTURE_PLATFORM: {
            'title': 'Infrastructure & Platform Engineer',
            'focus_areas': [
                'Container orchestration',
                'CI/CD pipelines',
                'Infrastructure as code',
                'Service mesh',
                'Observability'
            ],
            'relevant_problems': [
                'Resource scheduling',
                'Deployment strategies',
                'Service discovery',
                'Configuration management',
                'Auto-scaling'
            ],
            'key_metrics': ['resource utilization', 'deployment frequency', 'MTTR', 'infrastructure cost'],
            'typical_scenarios': [
                'Design auto-scaling systems',
                'Implement blue-green deployments',
                'Build service mesh routing',
                'Optimize container scheduling'
            ],
            'algorithm_preferences': ['bin packing', 'scheduling algorithms', 'graph algorithms', 'optimization algorithms']
        },
        
        RoleFamily.SECURITY_RELIABILITY: {
            'title': 'Security & Reliability Engineer',
            'focus_areas': [
                'Threat detection',
                'Incident response',
                'Chaos engineering',
                'Access control',
                'Compliance automation'
            ],
            'relevant_problems': [
                'Anomaly detection',
                'Rate limiting',
                'Circuit breaking',
                'Audit logging',
                'Encryption key management'
            ],
            'key_metrics': ['MTBF', 'detection accuracy', 'false positive rate', 'recovery time'],
            'typical_scenarios': [
                'Detect fraudulent patterns',
                'Implement circuit breakers',
                'Design audit systems',
                'Build DDoS protection'
            ],
            'algorithm_preferences': ['pattern matching', 'anomaly detection algorithms', 'cryptographic algorithms', 'statistical analysis']
        }
    }

    def __init__(self, provider_name: str = "claude", 
                 custom_config: Optional[Dict[str, Any]] = None):
        """Initialize the transformer with provider configuration"""
        # Initialize the LLM provider
        self.provider_name = provider_name
        self.llm_provider = get_provider(
            provider_name=provider_name,
            task_type=TaskType.CUSTOM_PROMPT_TRANSFORM,
            custom_config=custom_config
        )



    def extract_problem_keywords(self, problem: Problem) -> List[str]:
        """Extract keywords from problem description"""
        combined_text = f"{problem.title} {problem.description} {' '.join(problem.categories)}"
        text = combined_text.lower()
        keywords = []

        # Check for data structures
        for ds in self.DATA_STRUCTURES:
            if ds.lower() in text:
                keywords.append(ds)

        # Check for algorithms
        for algo in self.ALGORITHMS:
            if algo.lower() in text:
                keywords.append(algo)

        # Check for common terms
        common_terms = ['optimize', 'efficient', 'minimum', 'maximum', 'subarray', 'substring', 'consecutive']
        for term in common_terms:
            if term in text:
                keywords.append(term)

        # Add categories
        keywords.extend(problem.categories)

        # Check for complexity requirements
        if 'o(n)' in text:
            keywords.append('LinearTime')
        if 'o(1)' in text:
            keywords.append('ConstantSpace')
        if 'o(log n)' in text:
            keywords.append('Logarithmic')
        if 'o(n log n)' in text:
            keywords.append('LinearithmicTime')

        return list(set(keywords))  # Remove duplicates

    def detect_core_algorithms(self, problem: Problem) -> List[str]:
        """Detect core algorithms used in the problem"""
        combined_text = f"{problem.title} {problem.description} {' '.join(problem.categories)}"
        text = combined_text.lower()
        core_algorithms = []

        algorithm_patterns = [
            (r'(depth.first|dfs)', 'DepthFirstSearch'),
            (r'(breadth.first|bfs)', 'BreadthFirstSearch'),
            (r'(dynamic.programming|dp)', 'DynamicProgramming'),
            (r'(two.point|2.point)', 'TwoPointers'),
            (r'(sliding.window)', 'SlidingWindow'),
            (r'(binary.search)', 'BinarySearch'),
            (r'(backtrack)', 'Backtracking'),
            (r'(greedy)', 'Greedy'),
            (r'(recursion|recursive)', 'Recursion'),
            (r'(union.find|disjoint.set)', 'UnionFind'),
            (r'(topological.sort)', 'TopologicalSort'),
            (r'(hash|map|dictionary)', 'Hashing'),
            (r'(graph)', 'GraphAlgorithm'),
            (r'(tree)', 'TreeAlgorithm'),
            (r'(sort)', 'Sorting')
        ]

        for pattern, algorithm in algorithm_patterns:
            if re.search(pattern, text) and algorithm not in core_algorithms:
                core_algorithms.append(algorithm)

        # Check categories too
        for category in problem.categories:
            cat_lower = category.lower()
            for pattern, algorithm in algorithm_patterns:
                if re.search(pattern, cat_lower) and algorithm not in core_algorithms:
                    core_algorithms.append(algorithm)

        return core_algorithms

    def detect_data_structures(self, problem: Problem) -> List[str]:
        """Detect data structures used in the problem"""
        combined_text = f"{problem.title} {problem.description} {' '.join(problem.categories)}"
        text = combined_text.lower()
        data_structures = []

        structure_patterns = [
            (r'(array|arrays)', 'Array'),
            (r'(linked.list|linkedlist)', 'LinkedList'),
            (r'(stack)', 'Stack'),
            (r'(queue)', 'Queue'),
            (r'(hash.map|hashmap|hash.table|hashtable)', 'HashMap'),
            (r'(hash.set|hashset)', 'HashSet'),
            (r'(tree|trees)', 'Tree'),
            (r'(binary.tree)', 'BinaryTree'),
            (r'(binary.search.tree|bst)', 'BinarySearchTree'),
            (r'(heap)', 'Heap'),
            (r'(priority.queue)', 'PriorityQueue'),
            (r'(graph)', 'Graph'),
            (r'(trie)', 'Trie'),
            (r'(matrix|grid)', 'Matrix'),
            (r'(string)', 'String')
        ]

        for pattern, structure in structure_patterns:
            if re.search(pattern, text) and structure not in data_structures:
                data_structures.append(structure)

        # Check categories
        for category in problem.categories:
            cat_lower = category.lower()
            for pattern, structure in structure_patterns:
                if re.search(pattern, cat_lower) and structure not in data_structures:
                    data_structures.append(structure)

        return data_structures

    def extract_problem_info(self, problem: Problem) -> ExtractedProblemInfo:
        """Extract key information from a problem"""
        keywords = self.extract_problem_keywords(problem)
        core_algorithms = self.detect_core_algorithms(problem)
        data_structures = self.detect_data_structures(problem)

        # Extract sample test cases
        test_cases = [
            {"input": tc.stdin, "output": tc.expectedStdout}
            for tc in problem.test_cases if tc.is_sample
        ]

        return ExtractedProblemInfo(
            title=problem.title,
            difficulty=problem.difficulty,
            description=problem.description,
            constraints=problem.constraints,
            categories=problem.categories,
            time_complexity=problem.time_complexity,
            space_complexity=problem.space_complexity,
            keywords=keywords,
            core_algorithms=core_algorithms,
            data_structures=data_structures,
            default_user_code=problem.default_user_code,
            test_cases=test_cases
        )

    def extract_company_info(self, company: Company, problem_keywords: List[str]) -> ExtractedCompanyInfo:
        """Extract relevant company information for problem context"""
        relevant_keywords = []

        # Find relevant technologies
        for tech in company.technologies:
            for keyword in problem_keywords:
                if keyword.lower() in tech.lower() or tech.lower() in keyword.lower():
                    relevant_keywords.append(tech)

        # Find relevant products
        for product in company.products:
            product_lower = product.lower()
            for keyword in problem_keywords:
                if keyword.lower() in product_lower:
                    relevant_keywords.append(product)

        return ExtractedCompanyInfo(
            name=company.name,
            domain=company.domain,
            products=company.products,
            technologies=company.technologies,
            interview_focus=company.interview_focus,
            relevant_keywords=list(set(relevant_keywords))
        )

    def calculate_relevance_score(self, problem_info: ExtractedProblemInfo,
                                   company_info: ExtractedCompanyInfo) -> float:
        """Calculate relevance score between a problem and company"""
        score = 0.0

        # Check technology alignment
        for tech in company_info.technologies:
            tech_lower = tech.lower()
            for ds in problem_info.data_structures:
                if ds.lower() in tech_lower or tech_lower in ds.lower():
                    score += 2
            for algo in problem_info.core_algorithms:
                if algo.lower() in tech_lower or tech_lower in algo.lower():
                    score += 2

        # Check interview focus alignment
        for focus in company_info.interview_focus:
            focus_lower = focus.lower()
            if 'algorithm' in focus_lower and problem_info.core_algorithms:
                score += 2
            if 'data structure' in focus_lower and problem_info.data_structures:
                score += 2
            if 'system design' in focus_lower and problem_info.difficulty == ProblemDifficulty.HARD:
                score += 2
            if 'scale' in focus_lower and 'efficient' in problem_info.keywords:
                score += 2

        # Check domain relevance
        domain_lower = company_info.domain.lower()
        for keyword in problem_info.keywords:
            if keyword.lower() in domain_lower:
                score += 1

        # Add points for relevant keywords
        score += len(company_info.relevant_keywords)

        return score

    def generate_suggested_analogy_points(self, problem_info: ExtractedProblemInfo,
                                           company_info: ExtractedCompanyInfo,
                                           role_family: Optional[RoleFamily] = None) -> List[str]:
        """Generate suggested analogy points using enriched company data and role context."""
        analogy_points = []
        
        # Use company's embedded analogy patterns if available
        if hasattr(company_info, 'analogy_patterns') and company_info.analogy_patterns:
            for ds in problem_info.data_structures:
                if ds in company_info.analogy_patterns:
                    for pattern in company_info.analogy_patterns[ds]:
                        analogy = pattern.get('analogy', '') if isinstance(pattern, dict) else str(pattern)
                        analogy_points.append(analogy)
        
        # Use company's engineering challenges if available
        if hasattr(company_info, 'engineering_challenges') and company_info.engineering_challenges:
            for category, challenges in company_info.engineering_challenges.items():
                for challenge in challenges:
                    # Check if challenge matches problem characteristics
                    challenge_lower = challenge.lower()
                    for algo in problem_info.core_algorithms:
                        if algo.lower() in challenge_lower:
                            analogy_points.append(f"{company_info.name}'s {challenge}")
                            break
                    for ds in problem_info.data_structures:
                        if ds.lower() in challenge_lower:
                            analogy_points.append(f"{company_info.name}'s {challenge}")
                            break
        
        # Add role-specific analogies if role is specified
        if role_family and role_family in self.ROLE_CONFIGURATIONS:
            role_config = self.ROLE_CONFIGURATIONS[role_family]
            for scenario in role_config['typical_scenarios']:
                # Check if scenario matches problem characteristics
                scenario_lower = scenario.lower()
                for algo in problem_info.core_algorithms:
                    if algo.lower() in scenario_lower:
                        analogy_points.append(f"{company_info.name}'s {scenario.lower()}")
                        break
                for ds in problem_info.data_structures:
                    if ds.lower() in scenario_lower:
                        analogy_points.append(f"{company_info.name}'s {scenario.lower()}")
                        break
        
        # Use notable systems if available
        if hasattr(company_info, 'notable_systems') and company_info.notable_systems:
            for system in company_info.notable_systems[:2]:  # Limit to first 2
                analogy_points.append(f"{company_info.name}'s {system}")
        
        # Fallback to basic pattern matching if no enriched data
        if not analogy_points:
            # Use the original basic pattern matching
            data_structure_to_product_mappings = [
                {
                    'data_structure': 'Array',
                    'product_patterns': [
                        {'pattern': r'(search)', 'analogy': '{company} search results list'},
                        {'pattern': r'(video|stream)', 'analogy': '{company} video recommendations'},
                        {'pattern': r'(product|item|shop)', 'analogy': '{company} product catalog'}
                    ]
                },
                {
                    'data_structure': 'Graph',
                    'product_patterns': [
                        {'pattern': r'(map|navigation)', 'analogy': '{company} maps connections between locations'},
                        {'pattern': r'(social|network|connect)', 'analogy': '{company} user connections network'},
                        {'pattern': r'(recommendation)', 'analogy': '{company} recommendation system'}
                    ]
                },
                {
                    'data_structure': 'Tree',
                    'product_patterns': [
                        {'pattern': r'(file|document)', 'analogy': '{company} file/folder hierarchy'},
                        {'pattern': r'(category|taxonomy)', 'analogy': '{company} product category organization'},
                        {'pattern': r'(ui|interface)', 'analogy': '{company} UI component hierarchy'}
                    ]
                },
                {
                    'data_structure': 'HashMap',
                    'product_patterns': [
                        {'pattern': r'(cache|memory)', 'analogy': '{company} caching system'},
                        {'pattern': r'(user|profile)', 'analogy': '{company} user profile store'},
                        {'pattern': r'(config|setting)', 'analogy': '{company} configuration management'}
                    ]
                }
            ]
            
            for ds_info in data_structure_to_product_mappings:
                if ds_info['data_structure'] in problem_info.data_structures:
                    for product in company_info.products:
                        for p_pattern in ds_info['product_patterns']:
                            if re.search(p_pattern['pattern'], product.lower()):
                                analogy_points.append(p_pattern['analogy'].replace('{company}', company_info.name))
        
        # Final fallback
        if not analogy_points:
            analogy_points.append(
                f"{company_info.name}'s engineering challenges in the {company_info.domain} domain"
            )
        
        return list(set(analogy_points))  # Remove duplicates

    def create_transformation_context(self, problem: Problem,
                                       company: Company,
                                       role_family: Optional[RoleFamily] = None) -> TransformationContext:
        """Create transformation context for a problem and company with optional role"""
        problem_info = self.extract_problem_info(problem)
        company_info = self.extract_company_info(company, problem_info.keywords)
        relevance_score = self.calculate_relevance_score(problem_info, company_info)
        suggested_analogy_points = self.generate_suggested_analogy_points(
            problem_info, company_info, role_family
        )

        return TransformationContext(
            problem=problem_info,
            company=company_info,
            relevance_score=relevance_score,
            suggested_analogy_points=suggested_analogy_points
        )

    def generate_optimized_prompt(self, context: TransformationContext) -> str:
        """Generate an optimized prompt for transformation, matching the TS implementation."""
        problem = context.problem
        company = context.company

        # Extract function/class names to map
        function_names_to_map = ""
        if problem.default_user_code:
            func_matches = re.findall(r'def\s+(\w+)\s*\(', problem.default_user_code)
            class_matches = re.findall(r'class\s+(\w+)[:(]', problem.default_user_code)
            all_names = func_matches + class_matches

            if all_names:
                function_names_to_map = f"""
SPECIFIC FUNCTIONS/CLASSES TO RENAME:
{chr(10).join(all_names)}
Make sure your mapping includes ALL of these names from the original code.
"""

        # Format test cases as in the TS implementation
        test_cases_str = '\n'.join([f"* Input: {tc['input']}\n* Output: {tc['output']}" for tc in problem.test_cases])

        # Build the enhanced prompt from the TS implementation
        prompt = f"""
I need you to transform a coding problem into a company-specific interview scenario for {company.name}. This should feel like a real technical interview question.

ORIGINAL PROBLEM:
"{problem.title}" ({problem.difficulty.value})
{problem.description}

{f'''
ORIGINAL CODE TEMPLATE:
```python
{problem.default_user_code}
```
''' if problem.default_user_code else ''}

TECHNICAL ANALYSIS:
* Core Algorithms: {', '.join(problem.core_algorithms) or 'N/A'}
* Data Structures: {', '.join(problem.data_structures) or 'N/A'}
* Time Complexity: {problem.time_complexity or 'Not specified, but maintain the original complexity'}
* Space Complexity: {problem.space_complexity or 'Not specified, but maintain the original complexity'}

COMPANY CONTEXT:
* {company.name} specializes in: {company.domain}
* Key products: {', '.join(company.products)}
* Technology stack: {', '.join(company.technologies)}
* Interview focus areas: {', '.join(company.interview_focus)}

SUGGESTED COMPANY-SPECIFIC ANALOGIES:
{chr(10).join([f'* {point}' for point in context.suggested_analogy_points])}

{function_names_to_map}
YOUR TASK:
Create a realistic interview scenario that a {company.name} interviewer might present during a technical interview. The scenario MUST:

1. Maintain EXACTLY the same algorithmic approach and logical structure as the original
2. Preserve identical input/output data types (if input is an array of integers, keep it as an array of integers)
3. Include 2-3 clear examples with input→output test cases similar to the original problem. Use the test cases from the original problem {test_cases_str}
4. Frame the problem within {company.name}'s products, services, or technology domain
5. Keep the same time and space complexity requirements
6. Be concise, clear, and directly applicable to a technical interview setting
7. If applicable, mention any specific technical aspects of {company.name} that relate to the problem

IMPORTANT REQUIREMENTS:
* The core problem MUST remain mathematically and logically equivalent to the original
* The problem should sound natural, not forced - like a real interview question
* Include any constraints from the original problem (e.g., size limits, value ranges)
* Name any variables or functions in ways that align with {company.name}'s tech stack
* If the original has O(n) time complexity, maintain that exact requirement
* Make the scenario realistic - something a {company.name} engineer might actually work on

REQUIRED FUNCTION/CLASS MAPPING:
At the very end of your response, please include a clear mapping section that lists all the functions, classes, and variables you've renamed. Format it like this:

FUNCTION_MAPPING:
original_function_name -> new_function_name
original_class_name -> new_class_name
original_variable_name -> new_variable_name

FORMAT YOUR RESPONSE WITH THESE EXACT SECTIONS:
1. "# [COMPANY_NAME] Technical Interview Question: [PROBLEM_TITLE]" - Use a descriptive, specific title
2. "## Problem Background" - Context setting within company domain
3. "## The Problem" - Clear, concise problem statement (what needs to be solved)
4. "## Function Signature" - Code block with the function/class signature and docstring
5. "## Constraints" - List all constraints as bullet points
6. "## Examples" - 2-3 examples with input, output, and explanation
7. "## Requirements" - Time/space complexity and any other technical requirements

Format your response as a cohesive interview question, with an introduction, clear statement of the problem, and the function mapping section at the end.
DON'T CREATE MAPPING IF THE MAPPING IS UNCHANGED FOR SOME VALUE
"""
        return prompt

    def generate_optimized_prompt_with_role(self, context: TransformationContext, 
                                           role_family: Optional[RoleFamily] = None) -> str:
        """Generate prompt with role-specific context"""
        base_prompt = self.generate_optimized_prompt(context)
        
        if role_family and role_family in self.ROLE_CONFIGURATIONS:
            role_config = self.ROLE_CONFIGURATIONS[role_family]
            
            # Create role-specific context section
            role_section = f"""
ROLE-SPECIFIC CONTEXT:
* Position: {role_config['title']}
* Focus Areas: {', '.join(role_config['focus_areas'])}
* Relevant Problems: {', '.join(role_config['relevant_problems'][:3])}
* Key Metrics: {', '.join(role_config['key_metrics'])}
* Algorithm Preferences: {', '.join(role_config['algorithm_preferences'])}

ROLE-SPECIFIC REQUIREMENTS:
* Frame the problem in the context of a {role_config['title']} at {context.company.name}
* Use metrics and terminology specific to this role
* Consider typical scenarios this role would encounter
* Emphasize the scale and constraints relevant to this position
* Incorporate challenges from {', '.join(role_config['focus_areas'][:2])}
"""
            
            # Insert role section before "YOUR TASK:"
            base_prompt = base_prompt.replace(
                "YOUR TASK:",
                role_section + "\nYOUR TASK:"
            )
        
        return base_prompt

    

    def transform_problem(self, problem: Problem, company: Company, 
                          role_family: Optional[RoleFamily] = None) -> Dict[str, Any]:
        """Main function to transform a problem into a company-specific scenario with optional role"""

        # Create transformation context with role
        context = self.create_transformation_context(problem, company, role_family)

        # Generate optimized prompt with role context
        prompt = self.generate_optimized_prompt_with_role(context, role_family)

        # Log the task configuration (matching algo-irl logging)
        config_summary = self.llm_provider.get_config_summary()
        logger.info(f"Executing problem transformation using {config_summary['provider']} ({config_summary['model']})...")
        logger.info(f"Max output tokens: {config_summary['max_output_tokens']}")
        if config_summary['thinking_enabled']:
            logger.info(f"Thinking enabled: True (budget: {config_summary['thinking_budget']} tokens)")

        # Call LLM to generate the scenario
        try:
            scenario_text = self._call_llm_api(prompt)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Fallback to basic scenario if LLM fails
            scenario_text = self._generate_fallback_scenario(context)

        # Parse the scenario
        structured_scenario = parse_scenario_sections(scenario_text)

        # Prepare result
        result = {
            "structured_scenario": asdict(structured_scenario),
            "context_info": {
                "detected_algorithms": context.problem.core_algorithms,
                "detected_data_structures": context.problem.data_structures,
                "relevance_score": context.relevance_score,
                "suggested_analogy_points": context.suggested_analogy_points
            },
            "provider_info": {
                "provider": config_summary['provider'],
                "model": config_summary['model'],
                "timestamp": time.time()
            }
        }

        return result

    def _call_llm_api(self, prompt: str) -> str:
        """Call the configured LLM API to generate the scenario with retry logic"""
        # System prompt matching algo-irl
        system_prompt = """You are an expert technical interviewer who specializes in creating algorithm and data structure problems for software engineering interviews.
Your task is to transform coding problems into realistic company-specific interview scenarios while preserving their algorithmic essence.
Your scenarios should feel like actual questions a candidate would receive in a technical interview, with appropriate domain-specific framing that aligns with the company's business and technology."""

        # Use the provider's generate_completion method (which includes retry logic)
        return self.llm_provider.generate_completion(system_prompt, prompt)

    def _generate_fallback_scenario(self, context: TransformationContext) -> str:
        """Generate a fallback scenario when LLM API is unavailable"""
        problem = context.problem
        company = context.company

        # Create a basic transformed scenario as fallback
        scenario = f"""# {company.name} Technical Interview Question: Optimized {problem.title}

## Problem Background
At {company.name}, we're working on optimizing our {company.products[0] if company.products else 'system'}
to handle large-scale operations efficiently. Your task involves solving a critical
{problem.core_algorithms[0] if problem.core_algorithms else 'algorithmic'} challenge
that our engineering team faces daily.

## The Problem
{problem.description}

## Function Signature
```python
def optimize_{company.name.lower()}_system(data):
    \"\"\"
    Solve the {company.name} optimization problem.

    Args:
        data: Input data structure

    Returns:
        Optimized result
    \"\"\"
    pass
```

## Constraints
{chr(10).join([f'- {c}' for c in problem.constraints])}

## Examples
Example 1:
Input: {problem.test_cases[0]['input'] if problem.test_cases else 'sample_input_1'}
Output: {problem.test_cases[0]['output'] if problem.test_cases else 'sample_output_1'}
Explanation: This demonstrates the basic functionality.

{f'''Example 2:
Input: {problem.test_cases[1]['input']}
Output: {problem.test_cases[1]['output']}
Explanation: This shows edge case handling.''' if len(problem.test_cases) > 1 else ''}

## Requirements
- Time Complexity: {problem.time_complexity or 'O(n)'}
- Space Complexity: {problem.space_complexity or 'O(1)'}
- Must handle edge cases efficiently

FUNCTION_MAPPING:
solution -> optimize_{company.name.lower()}_system
twoSum -> find_{company.name.lower()}_pair
"""
        return scenario

    def test_llm_connection(self) -> bool:
        """Test if the LLM API connection is working"""
        return self.llm_provider.test_connection()

    def save_transformation_to_file(self, result: Dict[str, Any], problem_id: str, company_name: str):
        """Save transformation result to a provider-specific folder with unique filename including LLM config"""
        try:
            # Get LLM configuration details
            config_summary = self.llm_provider.get_config_summary()
            
            # Create provider-specific folder
            provider_folder = f"transformations/{self.provider_name}"
            os.makedirs(provider_folder, exist_ok=True)
            
            # Generate unique filename with LLM configuration
            # Sanitize names for filesystem compatibility
            safe_problem_id = re.sub(r'[<>:"/\\|?*]', '_', problem_id)
            safe_company_name = re.sub(r'[<>:"/\\|?*&]', '_', company_name.replace(' ', '_'))
            
            # Extract key configuration details for filename
            model_name = config_summary.get('model', 'unknown').replace('/', '_').replace(':', '_')
            max_tokens = config_summary.get('max_output_tokens', 'default')
            thinking_budget = config_summary.get('thinking_budget', 'default')
            thinking_enabled = 'thinking' if config_summary.get('thinking_enabled', False) else 'no-thinking'
            
            # Build filename with configuration details (no timestamp)
            filename = f"{safe_problem_id}_{safe_company_name}_{model_name}_{thinking_enabled}_{max_tokens}tok_{thinking_budget}budget.json"
            filepath = os.path.join(provider_folder, filename)
            
            # Save the transformation result
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Transformation saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save transformation: {e}")
            return None

    def list_transformations(self, problem_id: str, company_name: str) -> List[str]:
        """List all transformation files for a given problem and company"""
        try:
            provider_folder = f"transformations/{self.provider_name}"
            if not os.path.exists(provider_folder):
                return []
            
            # Sanitize names for pattern matching
            safe_problem_id = re.sub(r'[<>:"/\\|?*]', '_', problem_id)
            safe_company_name = re.sub(r'[<>:"/\\|?*&]', '_', company_name.replace(' ', '_'))
            
            # Pattern to match files for this problem and company
            pattern = f"{safe_problem_id}_{safe_company_name}_"
            
            matching_files = []
            for filename in os.listdir(provider_folder):
                if filename.startswith(pattern) and filename.endswith('.json'):
                    matching_files.append(os.path.join(provider_folder, filename))
            
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return matching_files
        except Exception as e:
            logger.error(f"Failed to list transformations: {e}")
            return []


def fetch_company_by_name(name: str) -> Optional[Company]:
    """Fetch a company from Firestore by its name."""
    if not db:
        logger.error("Firestore is not available.")
        return None
    
    from google.cloud.firestore_v1.base_query import FieldFilter
    companies_ref = db.collection('companies')
    query = companies_ref.where(filter=FieldFilter('name', '==', name)).limit(1)
    results = list(query.stream())
    
    if not results:
        logger.error(f"Company '{name}' not found.")
        return None
        
    company_doc = results[0]
    company_data = company_doc.to_dict()
    company_data['id'] = company_doc.id
    
    # Map firestore data to Company dataclass
    return Company(
        id=company_data.get('id'),
        name=company_data.get('name'),
        domain=company_data.get('domain'),
        products=company_data.get('products', []),
        technologies=company_data.get('technologies', []),
        interview_focus=company_data.get('interviewFocus', [])
    )

def fetch_problem_by_id(problem_id: str) -> Optional[Problem]:
    """Fetch a problem from Firestore by its ID."""
    if not db:
        logger.error("Firestore is not available.")
        return None
        
    problem_ref = db.collection('problems').document(problem_id)
    problem_doc = problem_ref.get()
    
    if not problem_doc.exists:
        logger.error(f"Problem with ID '{problem_id}' not found.")
        return None
        
    problem_data = problem_doc.to_dict()
    problem_data['id'] = problem_doc.id

    test_cases_data = problem_data.get('testCases', [])
    test_cases = []
    for tc_data in test_cases_data:
        # Filter out unexpected fields before creating the TestCase instance
        expected_fields = {f.name for f in fields(TestCase)}
        filtered_data = {k: v for k, v in tc_data.items() if k in expected_fields or k in ['input', 'output']}
        test_cases.append(TestCase(**filtered_data))

    # Map Firestore data to Problem dataclass
    return Problem(
        id=problem_data.get('id'),
        title=problem_data.get('title'),
        difficulty=ProblemDifficulty(problem_data.get('difficulty')),
        description=problem_data.get('description'),
        constraints=problem_data.get('constraints', []),
        categories=problem_data.get('categories', []),
        test_cases=test_cases,
        time_complexity=problem_data.get('timeComplexity'),
        space_complexity=problem_data.get('spaceComplexity'),
        default_user_code=problem_data.get('languageSpecificDetails', {}).get('python', {}).get('defaultUserCode', '')
    )


def main():
    """Main function with local caching and role support"""

    parser = argparse.ArgumentParser(
        description="Transform a coding problem into a company and role-specific scenario."
    )
    parser.add_argument("company_name", type=str, help="The name of the company")
    parser.add_argument("problem_id", type=str, help="The ID of the problem")
    parser.add_argument(
        "--provider",
        type=str,
        default="claude",
        choices=list_available_providers(),
        help="The LLM provider to use",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=['backend', 'ml', 'frontend', 'infrastructure', 'security'],
        help="The engineering role type for the transformation"
    )
    parser.add_argument(
        "--use-firestore",
        action="store_true",
        help="Use Firestore instead of local cache for company data"
    )

    args = parser.parse_args()
    
    # Map role string to RoleFamily enum
    role_family = None
    if args.role:
        role_map = {
            'backend': RoleFamily.BACKEND_SYSTEMS,
            'ml': RoleFamily.ML_DATA,
            'frontend': RoleFamily.FRONTEND_FULLSTACK,
            'infrastructure': RoleFamily.INFRASTRUCTURE_PLATFORM,
            'security': RoleFamily.SECURITY_RELIABILITY
        }
        role_family = role_map.get(args.role)

    # Initialize transformer with selected provider
    try:
        transformer = ProblemTransformer(provider_name=args.provider)
    except Exception as e:
        logger.error(f"Failed to initialize {args.provider} provider: {e}")
        print(f"\nTroubleshooting for {args.provider} provider:")
        if args.provider == "claude":
            print("1. Install Anthropic: pip install anthropic")
            print("2. Set API key: export ANTHROPIC_API_KEY='your-api-key'")
        elif args.provider == "gpt":
            print("1. Install OpenAI: pip install openai")
            print("2. Set API key: export OPENAI_API_KEY='your-api-key'")
        elif args.provider == "gemini":
            print("1. Install Google AI: pip install google-generativeai")
            print("2. Set API key: export GOOGLE_API_KEY='your-api-key'")
        print(f"\nAvailable providers: {', '.join(list_available_providers())}")
        return
    
    # Get company data (prefer local cache unless --use-firestore is specified)
    company = None
    if args.use_firestore and FIREBASE_AVAILABLE:
        company = fetch_company_by_name(args.company_name)
    else:
        # Use local cache with the same provider
        cache = CompanyCache(provider_name=args.provider)
        company = cache.get_company(args.company_name)
    
    # Fetch problem data (still from Firestore if available, otherwise create mock)
    problem = None
    if FIREBASE_AVAILABLE:
        problem = fetch_problem_by_id(args.problem_id)
    else:
        # Create a mock problem for testing without Firestore
        logger.warning("Firestore not available, using mock problem data")
        problem = Problem(
            id=args.problem_id,
            title="Two Sum",
            difficulty=ProblemDifficulty.EASY,
            description="Given an array of integers and a target sum, return indices of two numbers that add up to the target.",
            constraints=["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9"],
            categories=["Array", "Hash Table"],
            test_cases=[],
            time_complexity="O(n)",
            space_complexity="O(n)"
        )

    if not company or not problem:
        logger.error("Failed to fetch required data. Exiting.")
        return

    print("="*70)
    print("ALGO-IRL PROBLEM TRANSFORMER")
    print("="*70)
    print(f"\nTransforming problem '{problem.title}' for company '{company.name}'")
    if role_family:
        role_config = transformer.ROLE_CONFIGURATIONS[role_family]
        print(f"Role: {role_config['title']}")

    print("\n" + "="*70)
    print(f"Using '{args.provider}' provider for problem transformation")
    print("="*70)

    # Initialize transformer with selected provider
    try:
        transformer = ProblemTransformer(provider_name=args.provider)
        config_summary = transformer.llm_provider.get_config_summary()
        print(f"  Provider: {config_summary['provider']}")
        print(f"  Model: {config_summary['model']}")
        print(f"  Max Output Tokens: {config_summary['max_output_tokens']}")
        print(f"  Thinking: {config_summary['thinking_enabled']} (budget: {config_summary['thinking_budget']} tokens)")
        print(f"  API Key Found: {config_summary['has_api_key']}")
        print("="*70)
    except Exception as e:
        logger.error(f"Failed to initialize {args.provider} provider: {e}")
        print(f"\nTroubleshooting for {args.provider} provider:")
        if args.provider == "claude":
            print("1. Install Anthropic: pip install anthropic")
            print("2. Set API key: export ANTHROPIC_API_KEY='your-api-key'")
        elif args.provider == "gpt":
            print("1. Install OpenAI: pip install openai")
            print("2. Set API key: export OPENAI_API_KEY='your-api-key'")
        elif args.provider == "gemini":
            print("1. Install Google AI: pip install google-generativeai")
            print("2. Set API key: export GOOGLE_API_KEY='your-api-key'")
        
        # Show available providers
        print(f"\nAvailable providers: {', '.join(list_available_providers())}")
        print("Try a different provider with --provider <name>")
        return

    # Test connection
    print("\nTesting LLM connection...")
    if transformer.test_llm_connection():
        print("✅ LLM connection successful!")
    else:
        print("❌ LLM connection failed. Using fallback scenario generator.")

    try:
        # Show existing transformations for reference (if any)
        existing_files = transformer.list_transformations(problem.id, company.name)
        if existing_files:
            print(f"\n📁 Existing transformations for '{problem.title}' and {company.name}:")
            for i, filepath in enumerate(existing_files[:5], 1):  # Show up to 5 most recent
                filename = os.path.basename(filepath)
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(filepath)))
                print(f"  {i}. {filename} (modified: {mod_time})")
            if len(existing_files) > 5:
                print(f"  ... and {len(existing_files) - 5} more")
        
        # Always generate new transformation for comparison
        print(f"\nGenerating new transformation for '{problem.title}' and {company.name}...")
        result = transformer.transform_problem(problem, company, role_family)

        # Print results
        print("\n" + "="*70)
        print("TRANSFORMED SCENARIO:")
        print("="*70)
        scenario = result['structured_scenario']
        print(f"\n📝 Title: {scenario['title']}")
        print(f"\n🏢 Background:\n{scenario['background']}")
        print(f"\n❓ Problem Statement:\n{scenario['problem_statement']}")
        print(f"\n💻 Function Signature:\n{scenario['function_signature']}")
        print(f"\n⚠️  Constraints:")
        for constraint in scenario['constraints']:
            print(f"  • {constraint}")
        print(f"\n📊 Examples:")
        for i, example in enumerate(scenario['examples'], 1):
            print(f"  Example {i}:")
            print(f"    Input: {example['input']}")
            print(f"    Output: {example['output']}")
            if example.get('explanation'):
                print(f"    Explanation: {example['explanation']}")

        # Show context info
        print("\n" + "="*70)
        print("TRANSFORMATION CONTEXT INFO:")
        print("="*70)
        print(f"🔍 Detected Algorithms: {', '.join(result['context_info']['detected_algorithms'])}")
        print(f"📚 Detected Data Structures: {', '.join(result['context_info']['detected_data_structures'])}")
        print(f"⭐ Relevance Score: {result['context_info']['relevance_score']}")
        print(f"💡 Suggested Analogies:")
        for analogy in result['context_info']['suggested_analogy_points']:
            print(f"  • {analogy}")

        # Show function mappings if any
        if scenario.get('function_mapping'):
            print("\n" + "="*70)
            print("FUNCTION MAPPINGS:")
            print("="*70)
            for original, renamed in scenario['function_mapping'].items():
                print(f"  {original} → {renamed}")

        # Save transformation result to provider-specific folder
        # Include role in filename if specified
        filename_company = f"{company.name}_{role_family.value}" if role_family else company.name
        saved_filepath = transformer.save_transformation_to_file(result, problem.id, filename_company)
        if saved_filepath:
            print(f"\n💾 Transformation saved to {saved_filepath}")
        else:
            print(f"\n⚠️ Failed to save transformation to file")

    except Exception as e:
        logger.error(f"An error occurred during transformation: {e}", exc_info=True)
        print("\n" + "="*70)
        print("⚠️  Provider Setup Guide:")
        print("="*70)
        print("Claude (Anthropic):")
        print("  1. Install: pip install anthropic")
        print("  2. Set API key: export ANTHROPIC_API_KEY='your-api-key'")
        print("  3. Usage: --provider claude")
        print("\nGPT (OpenAI):")
        print("  1. Install: pip install openai")
        print("  2. Set API key: export OPENAI_API_KEY='your-api-key'")
        print("  3. Usage: --provider gpt")
        print("\nGemini (Google):")
        print("  1. Install: pip install google-generativeai")
        print("  2. Set API key: export GOOGLE_API_KEY='your-api-key'")
        print("  3. Usage: --provider gemini")


if __name__ == "__main__":
    main()
