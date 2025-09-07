# Problem Transformer Usage Guide

## Overview
The enhanced problem transformer now supports:
- **Role-based transformations** for 5 engineering role types
- **Local company data caching** to reduce API calls
- **Optimized model selection** for company data generation

## Quick Start

### Basic Usage
```bash
# Transform a problem for a company (uses local cache)
python -m transform_generation.problem_transformer Google two-sum
```

### With Role Specification
Create role-specific interview scenarios:

```bash
# Backend & Distributed Systems Engineer
python -m transform_generation.problem_transformer Google two-sum --role backend

# Machine Learning & Data Engineer
python -m transform_generation.problem_transformer Meta graph-traversal --role ml

# Frontend & Full-Stack Engineer
python -m transform_generation.problem_transformer Airbnb search-ui --role frontend

# Infrastructure & Platform Engineer
python -m transform_generation.problem_transformer Amazon two-sum --role infrastructure

# Security & Reliability Engineer
python -m transform_generation.problem_transformer Stripe payment-processing --role security
```

### With Different Providers
```bash
# Use GPT-4 (OpenAI)
python -m transform_generation.problem_transformer Google two-sum --provider gpt

# Use Gemini (Google)
python -m transform_generation.problem_transformer Google two-sum --provider gemini

# Use Claude (default)
python -m transform_generation.problem_transformer Google two-sum --provider claude
```

### Combined Options
```bash
# Specific provider + role
python -m transform_generation.problem_transformer Uber ride-matching --provider gpt --role backend

# All options
python -m transform_generation.problem_transformer Netflix video-streaming --provider gemini --role infrastructure
```

## Role Descriptions

| Role | Focus Areas | Key Metrics |
|------|-------------|-------------|
| **backend** | APIs, distributed systems, databases, microservices | RPS, p99 latency, throughput |
| **ml** | Data pipelines, feature engineering, model serving | Data freshness, pipeline latency |
| **frontend** | UI performance, state management, real-time updates | FCP, TTI, bundle size |
| **infrastructure** | Container orchestration, CI/CD, service mesh | Resource utilization, MTTR |
| **security** | Threat detection, incident response, compliance | MTBF, detection accuracy |

## Features

### Local Company Caching
- Companies are cached in `company_data_cache/` directory
- First run generates comprehensive company data
- Subsequent runs use cached data (instant)
- Cache includes enhanced context:
  - Engineering challenges
  - Scale metrics
  - Tech stack layers
  - Notable systems
  - Company-specific analogies

### Optimized Model Selection
Company data generation uses cost-effective models:
- **Claude**: `claude-3-5-haiku` (fast, efficient)
- **GPT**: `gpt-3.5-turbo` (cost-effective)
- **Gemini**: `gemini-1.5-flash` (rapid generation)

Problem transformation uses powerful models:
- **Claude**: `claude-sonnet-4` with thinking
- **GPT**: `gpt-5` with reasoning
- **Gemini**: `gemini-2.5-pro` with thinking

## Output

Transformations are saved to:
```
transformations/
├── claude/
│   ├── two-sum_Google_backend_claude-sonnet-4_thinking_15000tok_10000budget.json
│   └── ...
├── gpt/
│   └── ...
└── gemini/
    └── ...
```

## Environment Setup

### Required API Keys
```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY='your-api-key'

# For GPT (OpenAI)
export OPENAI_API_KEY='your-api-key'

# For Gemini (Google)
export GOOGLE_API_KEY='your-api-key'
```

### Installation
```bash
# Core dependencies
pip install anthropic openai google-generativeai

# Optional (for Firestore)
pip install firebase-admin

# For environment variables
pip install python-dotenv
```

## Advanced Options

### Use Firestore Instead of Local Cache
```bash
# Requires Firebase setup
python -m transform_generation.problem_transformer Google two-sum --use-firestore
```

### Clear Company Cache
```bash
# Remove cached company data
rm -rf company_data_cache/
```

### View Help
```bash
python -m transform_generation.problem_transformer --help
```

## Examples by Use Case

### Interview Prep for Specific Role
```bash
# Preparing for backend role at Google
python -m transform_generation.problem_transformer Google two-sum --role backend
python -m transform_generation.problem_transformer Google lru-cache --role backend
python -m transform_generation.problem_transformer Google rate-limiter --role backend
```

### Compare Different Company Contexts
```bash
# Same problem, different companies
python -m transform_generation.problem_transformer Google two-sum
python -m transform_generation.problem_transformer Meta two-sum
python -m transform_generation.problem_transformer Amazon two-sum
```

### Role-Specific Problem Sets
```bash
# ML/Data role problems
python -m transform_generation.problem_transformer Netflix recommendation --role ml
python -m transform_generation.problem_transformer Spotify playlist-generation --role ml
python -m transform_generation.problem_transformer TikTok feed-ranking --role ml
```

## Troubleshooting

### API Key Not Found
```
Error: No ANTHROPIC_API_KEY found
Solution: export ANTHROPIC_API_KEY='your-key'
```

### Company Not Generating
- Check API key is set correctly
- Verify internet connection
- Check `company_data_cache/companies.json` for errors

### Firestore Not Available
- This is normal if not using Firebase
- The script will use mock problem data for testing
- Use local cache mode (default) instead

## Best Practices

1. **First Run**: Let company data generate and cache (takes ~5-10 seconds)
2. **Role Selection**: Choose role based on target position
3. **Provider Choice**: 
   - Claude: Best for complex reasoning
   - GPT: Good balance of speed and quality
   - Gemini: Fastest generation
4. **Cache Management**: Keep cache for frequently used companies
5. **Problem Selection**: Use problem IDs from your problem database
