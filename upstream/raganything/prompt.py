"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# JSON formatting instructions (shared across all analysis prompts)
JSON_FORMAT_INSTRUCTIONS = """
CRITICAL JSON FORMATTING RULES:
1. Return ONLY valid JSON - no markdown code blocks, no backticks, no explanations outside JSON
2. All backslashes must be escaped: use \\\\ instead of \\
3. For LaTeX: \\\\alpha, \\\\beta, \\\\frac{{a}}{{b}}, etc.
4. All quotes inside strings must be escaped: use \\" for literal quotes
5. No trailing commas after the last item in objects or arrays
6. No line breaks inside string values - use \\n for newlines
7. Start your response with {{ and end with }}"""

# System prompts for different analysis types
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed, accurate descriptions. "
    "Always respond with valid JSON only - no markdown, no code blocks."
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed analysis based on available information. "
    "Always respond with valid JSON only - no markdown, no code blocks."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "You are an expert data analyst. Provide detailed table analysis with specific insights. "
    "Always respond with valid JSON only - no markdown, no code blocks."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "You are an expert mathematician. Provide detailed mathematical analysis. "
    "Always respond with valid JSON only - no markdown, no code blocks. "
    "Remember to double-escape all LaTeX backslashes in JSON strings."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "You are an expert content analyst specializing in {content_type} content. "
    "Always respond with valid JSON only - no markdown, no code blocks."
)

# Image analysis prompt template
PROMPTS[
    "vision_prompt"
] = """Analyze this image and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive visual description here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Describe overall composition and layout
- Identify all objects, people, text, and visual elements
- Explain relationships between elements
- Note colors, lighting, and visual style
- Describe any actions or activities shown
- Include technical details if relevant (charts, diagrams, etc.)
- Always use specific names instead of pronouns

Image Information:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}
""" + JSON_FORMAT_INSTRUCTIONS

# Image analysis prompt with context support
PROMPTS[
    "vision_prompt_with_context"
] = """Analyze this image considering the surrounding context and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive visual description with context references here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary with context relationship (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Describe overall composition and layout
- Identify all objects, people, text, and visual elements
- Explain relationships between elements and surrounding context
- Note colors, lighting, and visual style
- Describe any actions or activities shown
- Include technical details if relevant (charts, diagrams, etc.)
- Reference connections to the surrounding content
- Always use specific names instead of pronouns

Context from surrounding content:
{context}

Image Information:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}
""" + JSON_FORMAT_INSTRUCTIONS

# Image analysis prompt with text fallback
PROMPTS["text_prompt"] = """Based on the following image information, provide analysis:

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}"""

# Table analysis prompt template
PROMPTS[
    "table_prompt"
] = """Analyze this table and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive table analysis here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of table purpose and findings (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Table structure and organization
- Column headers and their meanings
- Key data points and patterns
- Statistical insights and trends
- Relationships between data elements
- Significance of the data presented
- Always use specific names and values instead of general references

Table Information:
- Image Path: {table_img_path}
- Caption: {table_caption}
- Body: {table_body}
- Footnotes: {table_footnote}
""" + JSON_FORMAT_INSTRUCTIONS

# Table analysis prompt with context support
PROMPTS[
    "table_prompt_with_context"
] = """Analyze this table considering the surrounding context and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive table analysis with context references here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary with context relationship (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Table structure and organization
- Column headers and their meanings
- Key data points and patterns
- Statistical insights and trends
- Relationships between data elements
- Significance in relation to surrounding context
- How the table supports or illustrates concepts from surrounding content
- Always use specific names and values instead of general references

Context from surrounding content:
{context}

Table Information:
- Image Path: {table_img_path}
- Caption: {table_caption}
- Body: {table_body}
- Footnotes: {table_footnote}
""" + JSON_FORMAT_INSTRUCTIONS

# Equation analysis prompt template
PROMPTS[
    "equation_prompt"
] = """Analyze this mathematical equation and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive equation analysis here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of equation purpose (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Mathematical meaning and interpretation
- Variables and their definitions
- Mathematical operations and functions used
- Application domain and context
- Physical or theoretical significance
- Relationship to other mathematical concepts
- Practical applications or use cases
- Always use specific mathematical terminology

Equation Information:
- Equation: {equation_text}
- Format: {equation_format}

IMPORTANT: When writing LaTeX in JSON strings, double-escape all backslashes.
Example: Write \\\\alpha instead of \\alpha, \\\\frac{{a}}{{b}} instead of \\frac{{a}}{{b}}
""" + JSON_FORMAT_INSTRUCTIONS

# Equation analysis prompt with context support
PROMPTS[
    "equation_prompt_with_context"
] = """Analyze this mathematical equation considering the surrounding context and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive equation analysis with context references here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary with context relationship (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Mathematical meaning and interpretation
- Variables and their definitions in context
- Mathematical operations and functions used
- Application domain based on surrounding material
- Physical or theoretical significance
- Relationship to other mathematical concepts in context
- Practical applications or use cases
- How the equation relates to the broader discussion
- Always use specific mathematical terminology

Context from surrounding content:
{context}

Equation Information:
- Equation: {equation_text}
- Format: {equation_format}

IMPORTANT: When writing LaTeX in JSON strings, double-escape all backslashes.
Example: Write \\\\alpha instead of \\alpha, \\\\frac{{a}}{{b}} instead of \\frac{{a}}{{b}}
""" + JSON_FORMAT_INSTRUCTIONS

# Generic content analysis prompt template
PROMPTS[
    "generic_prompt"
] = """Analyze this {content_type} content and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive content analysis here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of content purpose (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Content structure and organization
- Key information and elements
- Relationships between components
- Context and significance
- Relevant details for knowledge retrieval
- Always use specific terminology appropriate for {content_type} content

Content: {content}
""" + JSON_FORMAT_INSTRUCTIONS

# Generic content analysis prompt with context support
PROMPTS[
    "generic_prompt_with_context"
] = """Analyze this {content_type} content considering the surrounding context and return a JSON object with this exact structure:

{{
    "detailed_description": "comprehensive content analysis with context references here",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary with context relationship (max 100 words)"
    }}
}}

Guidelines for detailed_description:
- Content structure and organization
- Key information and elements
- Relationships between components
- Context and significance in relation to surrounding content
- How this content connects to or supports the broader discussion
- Relevant details for knowledge retrieval
- Always use specific terminology appropriate for {content_type} content

Context from surrounding content:
{context}

Content: {content}
""" + JSON_FORMAT_INSTRUCTIONS

# Modal chunk templates
PROMPTS["image_chunk"] = """
Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""

# Query-related prompts
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Please briefly describe the main content, key elements, and important information in this image."
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "You are a professional image analyst who can accurately describe image content."
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "You are a professional data analyst who can accurately analyze table data."
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "You are a mathematics expert who can clearly explain mathematical formulas."
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "You are a professional content analyst who can accurately analyze {content_type} type content."
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."
)
