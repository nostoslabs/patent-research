"""Multi-provider LLM factory using PydanticAI for patent similarity evaluation."""

import os
from enum import Enum
from typing import Optional, Union, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class PatentSimilarityAnalysis(BaseModel):
    """Structured output for patent similarity evaluation."""
    similarity_score: float = Field(
        ge=0, le=1, 
        description="Overall similarity score between 0 and 1"
    )
    technical_field_match: float = Field(
        ge=0, le=1, 
        description="How similar are the technical fields (0=different, 1=identical)"
    )
    problem_similarity: float = Field(
        ge=0, le=1, 
        description="How similar are the problems being solved"
    )
    solution_similarity: float = Field(
        ge=0, le=1, 
        description="How similar are the proposed solutions"
    )
    explanation: str = Field(
        description="Detailed explanation of similarity assessment"
    )
    key_concepts_1: List[str] = Field(
        description="Key technical concepts from patent 1"
    )
    key_concepts_2: List[str] = Field(
        description="Key technical concepts from patent 2"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Confidence in this assessment"
    )


class LLMFactory:
    """Factory for creating LLM agents with different providers."""
    
    @staticmethod
    def get_available_providers() -> List[LLMProvider]:
        """Get list of available providers based on API keys."""
        available = []
        
        if os.getenv("OPENAI_API_KEY"):
            available.append(LLMProvider.OPENAI)
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            available.append(LLMProvider.GOOGLE)
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append(LLMProvider.ANTHROPIC)
        
        # Ollama is always available if service is running
        available.append(LLMProvider.OLLAMA)
        
        return available
    
    @staticmethod
    def create_agent(provider: Optional[LLMProvider] = None) -> Agent:
        """Create an agent based on available API keys and preference."""
        
        if provider is None:
            # Auto-select best available provider
            available = LLMFactory.get_available_providers()
            if LLMProvider.GOOGLE in available:
                provider = LLMProvider.GOOGLE  # Gemini Flash is fastest and cheapest
            elif LLMProvider.OPENAI in available:
                provider = LLMProvider.OPENAI
            elif LLMProvider.ANTHROPIC in available:
                provider = LLMProvider.ANTHROPIC
            else:
                provider = LLMProvider.OLLAMA
        
        # Create model based on provider
        if provider == LLMProvider.OPENAI and os.getenv("OPENAI_API_KEY"):
            model = 'openai:gpt-4o-mini'  # Cost-effective for bulk processing
            
        elif provider == LLMProvider.GOOGLE and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            model = 'gemini-1.5-flash'  # Fast and efficient
            
        elif provider == LLMProvider.ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
            model = 'claude-3-haiku'  # Fast and cost-effective
            
        elif provider == LLMProvider.OLLAMA:
            # Use whatever Ollama model is available
            try:
                import ollama
                models = [m['name'] for m in ollama.list()['models']]
                if models:
                    # Prefer smaller, faster models for bulk processing
                    preferred = ['llama3.2:1b', 'llama3.2:3b', 'llama3.2', 'mistral']
                    model = None
                    for pref in preferred:
                        if any(pref in m for m in models):
                            model = f'ollama:{pref}'
                            break
                    if not model:
                        model = f'ollama:{models[0]}'  # Use first available
                else:
                    raise ValueError("No Ollama models available")
            except Exception:
                raise ValueError("Ollama service not available")
        else:
            raise ValueError(f"Provider {provider} not available or no API key found")
        
        system_prompt = """You are an expert patent examiner with deep technical knowledge across multiple fields. Your task is to evaluate the similarity between two patent abstracts.

Consider these aspects:
1. **Technical Field**: Are they in the same or related technical domains?
2. **Problem Solved**: Do they address similar problems or challenges?
3. **Solution Approach**: Are the proposed solutions similar in method or outcome?
4. **Technical Depth**: Consider the technical sophistication and novelty

Provide a thorough analysis with specific evidence from the abstracts. Be precise in your similarity scoring:
- 0.9-1.0: Nearly identical inventions (potential interference)
- 0.7-0.9: Very similar inventions in same field with similar solutions
- 0.5-0.7: Related inventions with some common elements
- 0.3-0.5: Loosely related inventions in similar fields
- 0.0-0.3: Different inventions with minimal similarity

Extract key technical concepts and provide detailed reasoning for your assessment."""

        return Agent(
            model,
            output_type=PatentSimilarityAnalysis,
            system_prompt=system_prompt
        )
    
    @staticmethod
    def estimate_cost(provider: LLMProvider, num_comparisons: int, avg_chars_per_patent: int = 3000) -> dict:
        """Estimate costs for different providers."""
        # Rough cost estimates per 1M tokens (input/output)
        costs = {
            LLMProvider.OPENAI: {"input": 0.15, "output": 0.60},  # gpt-4o-mini
            LLMProvider.GOOGLE: {"input": 0.075, "output": 0.30},  # gemini-1.5-flash
            LLMProvider.ANTHROPIC: {"input": 0.25, "output": 1.25},  # claude-3-haiku
            LLMProvider.OLLAMA: {"input": 0.0, "output": 0.0}  # Free local
        }
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        chars_per_comparison = avg_chars_per_patent * 2  # Two patents
        tokens_per_comparison = chars_per_comparison / 4
        
        total_input_tokens = (tokens_per_comparison * num_comparisons) / 1_000_000
        total_output_tokens = (200 * num_comparisons) / 1_000_000  # ~200 tokens output
        
        provider_costs = costs.get(provider, {"input": 0, "output": 0})
        
        total_cost = (total_input_tokens * provider_costs["input"] + 
                     total_output_tokens * provider_costs["output"])
        
        return {
            "provider": provider.value,
            "total_comparisons": num_comparisons,
            "estimated_input_tokens": int(total_input_tokens * 1_000_000),
            "estimated_output_tokens": int(total_output_tokens * 1_000_000),
            "estimated_cost_usd": round(total_cost, 3)
        }


def test_factory():
    """Test the LLM factory with available providers."""
    print("Available providers:")
    for provider in LLMFactory.get_available_providers():
        print(f"  - {provider.value}")
        
        # Test cost estimation
        cost_est = LLMFactory.estimate_cost(provider, 500, 3000)
        print(f"    Estimated cost for 500 comparisons: ${cost_est['estimated_cost_usd']}")
    
    print("\nTesting agent creation...")
    try:
        agent = LLMFactory.create_agent()
        print(f"✅ Successfully created agent")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None


if __name__ == "__main__":
    test_factory()