import re
from typing import Dict, List, Any
from .models import StructuredScenario

def _extract_title(scenario_text: str) -> str:
    match = re.search(r"^#\s+(.+?)(?:\n|$)", scenario_text, re.MULTILINE)
    return match.group(1).strip() if match else ""

def _extract_section(section_name: str, scenario_text: str, stop_at: str = "##") -> str:
    match = re.search(fr"## {section_name}\s*([\s\S]*?)(?={stop_at}|$)", scenario_text)
    return match.group(1).strip() if match else ""

def _extract_constraints(scenario_text: str) -> List[str]:
    constraints_text = _extract_section("Constraints", scenario_text)
    if not constraints_text:
        return []
    items = re.split(r'\n\s*[-*•]\s*|\n\s*\d+\.\s*', constraints_text)
    # Clean up items: remove empty strings and leading/trailing whitespace
    cleaned_items = [item.strip() for item in items if item.strip()]
    # Remove any remaining bullet points from the start of an item
    return [re.sub(r'^[-*•]\s*', '', item).strip() for item in cleaned_items]

def _extract_examples(scenario_text: str) -> List[Dict[str, Any]]:
    examples_text = _extract_section("Examples", scenario_text)
    if not examples_text:
        return []

    # Strategy 1: Look for "Example X:" blocks
    example_blocks = re.split(r'\s*(?:\*\*)?Example \d+(?:\*\*)?:?\s*', examples_text, flags=re.IGNORECASE)
    example_blocks = [block.strip() for block in example_blocks if block.strip()]

    if example_blocks:
        parsed_examples = []
        for block in example_blocks:
            # Sub-strategy: Handle bullet-point formatted examples
            if re.search(r'^\s*[\-\*•]\s*Input:', block, re.MULTILINE | re.IGNORECASE):
                input_match = re.search(r'\s*[\-\*•]\s*Input:?\s*([^\n]*(?:\n(?![\-\*•]\s*(?:Output|Explanation))[^\n]*)*)', block, re.IGNORECASE)
                output_match = re.search(r'\s*[\-\*•]\s*Output:?\s*([^\n]*(?:\n(?![\-\*•]\s*(?:Input|Explanation))[^\n]*)*)', block, re.IGNORECASE)
                explanation_match = re.search(r'\s*[\-\*•]\s*Explanation:?\s*([^\n]*(?:\n(?![\-\*•]\s*(?:Input|Output))[^\n]*)*)', block, re.IGNORECASE)
            # Sub-strategy: Standard format
            else:
                input_match = re.search(r'Input:?\s*([^\n]*(?:\n(?!Output:|Explanation:)[^\n]*)*)', block, re.IGNORECASE)
                output_match = re.search(r'Output:?\s*([^\n]*(?:\n(?!Input:|Explanation:)[^\n]*)*)', block, re.IGNORECASE)
                explanation_match = re.search(r'Explanation:?\s*([^\n]*(?:\n(?!Input:|Output:)[^\n]*)*)', block, re.IGNORECASE)

            parsed_examples.append({
                "input": input_match.group(1).strip() if input_match else "",
                "output": output_match.group(1).strip() if output_match else "",
                "explanation": explanation_match.group(1).strip() if explanation_match else None
            })
        return parsed_examples

    # Strategy 2 (Fallback): Find examples by looking for code blocks
    code_blocks = re.findall(r'```[^`]*```', examples_text)
    if len(code_blocks) >= 2:
        examples = []
        for i in range(0, len(code_blocks) - 1, 2):
            examples.append({
                "input": re.sub(r'```(?:python|javascript|java)?\n?|\n?```', '', code_blocks[i]).strip(),
                "output": re.sub(r'```(?:python|javascript|java)?\n?|\n?```', '', code_blocks[i+1]).strip()
            })
        return examples

    # Strategy 3 (Fallback): Simple Input/Output pattern matching
    inputs = [m.group(1).strip() for m in re.finditer(r'Input:?\s*([^\n]*(?:\n(?!Output:|Explanation:|Example \d+:)[^\n]*)*)', examples_text, re.IGNORECASE)]
    outputs = [m.group(1).strip() for m in re.finditer(r'Output:?\s*([^\n]*(?:\n(?!Input:|Explanation:|Example \d+:)[^\n]*)*)', examples_text, re.IGNORECASE)]
    explanations = [m.group(1).strip() for m in re.finditer(r'Explanation:?\s*([^\n]*(?:\n(?!Input:|Output:|Example \d+:)[^\n]*)*)', examples_text, re.IGNORECASE)]

    final_examples = []
    for i in range(min(len(inputs), len(outputs))):
        final_examples.append({
            "input": inputs[i],
            "output": outputs[i],
            "explanation": explanations[i] if i < len(explanations) else None
        })

    return final_examples

def _extract_requirements(scenario_text: str) -> List[str]:
    requirements_text = _extract_section("Requirements", scenario_text, stop_at="FUNCTION_MAPPING:")
    if not requirements_text:
        return []
    items = re.split(r'\n\s*[-*•]\s*|\n\s*\d+\.\s*', requirements_text)
    cleaned_items = [item.strip() for item in items if item.strip()]
    return [re.sub(r'^[-*•]\s*', '', item).strip() for item in cleaned_items]

def _extract_function_mapping(scenario_text: str) -> Dict[str, str]:
    function_mapping = {}
    match = re.search(r"FUNCTION_MAPPING:\s*([\s\S]+?)(?:\n\n|$)", scenario_text)
    if match and match.group(1):
        mapping_lines = match.group(1).strip().split('\n')
        for line in mapping_lines:
            if '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    original, renamed = parts
                    function_mapping[original.strip()] = renamed.strip()
    return function_mapping

def parse_scenario_sections(scenario_text: str) -> StructuredScenario:
    """Parse the scenario text into structured sections using robust helpers."""
    return StructuredScenario(
        title=_extract_title(scenario_text),
        background=_extract_section("Problem Background", scenario_text),
        problem_statement=_extract_section("The Problem", scenario_text),
        function_signature=_extract_section("Function Signature", scenario_text),
        constraints=_extract_constraints(scenario_text),
        examples=_extract_examples(scenario_text),
        requirements=_extract_requirements(scenario_text),
        function_mapping=_extract_function_mapping(scenario_text)
    )
