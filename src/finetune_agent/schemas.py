"""Pydantic schemas for the finetune agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class Difficulty(str, Enum):
    """Difficulty levels for Q&A pairs."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TrainingValue(str, Enum):
    """Estimated training value of a Q&A pair."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelFamily(str, Enum):
    """Target model family for the dataset."""
    CODE_LLM = "code_llm"
    CHAT_LLM = "chat_llm"
    CLASSIFIER = "classifier"
    INSTRUCT = "instruct"
    OTHER = "other"


# =============================================================================
# Core Data Models
# =============================================================================

class QAMetadata(BaseModel):
    """Metadata for a Q&A pair with V2 fields."""
    
    id: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    intent_label: str = ""
    estimated_training_value: TrainingValue = TrainingValue.MEDIUM
    source: Literal["synthetic", "template", "user"] = "synthetic"
    # Additional flexible metadata
    extra: dict[str, Any] = Field(default_factory=dict)


class QAPair(BaseModel):
    """A single question-answer pair with metadata."""
    
    question: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A dataset containing Q&A pairs of a specific type."""
    
    type: str
    items: list[QAPair]
    intents: list[str] = Field(default_factory=list)  # V2: tracked intents


class DatasetOutput(BaseModel):
    """Complete output from the dataset generator."""
    
    project_summary: str
    datasets: list[Dataset]
    generation_method: str = "template"  # V2: "template" or "llm"
    llm_provider: str = ""  # V2: which LLM was used


# =============================================================================
# Critique Models (V2)
# =============================================================================

class CritiqueResult(BaseModel):
    """Result from the self-critique agent."""
    
    reject_indices: list[int] = Field(default_factory=list)
    improvement_notes: list[str] = Field(default_factory=list)
    quality_assessment: str = "acceptable"
    duplicate_pairs: list[tuple[int, int]] = Field(default_factory=list)
    low_quality_indices: list[int] = Field(default_factory=list)


# =============================================================================
# Evaluation Models
# =============================================================================

class HealthMetrics(BaseModel):
    """Dataset health metrics for V2 evaluation."""
    
    avg_answer_length: float = 0.0
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    intent_coverage_score: float = 0.0
    items_with_code: int = 0
    items_with_code_pct: float = 0.0


class EvaluationResult(BaseModel):
    """Evaluation results for a dataset."""
    
    dataset_type: str
    uniqueness_score: float = Field(ge=0, le=100)
    item_count: int
    avg_question_length: float
    avg_answer_length: float
    # V2 additions
    lexical_score: float = Field(default=0.0, ge=0, le=100)
    structural_score: float = Field(default=0.0, ge=0, le=100)
    conceptual_score: float = Field(default=0.0, ge=0, le=100)
    health_metrics: HealthMetrics = Field(default_factory=HealthMetrics)


class OverallEvaluation(BaseModel):
    """Overall evaluation of all datasets."""
    
    dataset_evaluations: list[EvaluationResult]
    overall_rating: float = Field(ge=0, le=100)
    feedback: list[str]
    generated_at: datetime = Field(default_factory=datetime.now)
    # V2 additions
    health_metrics: HealthMetrics = Field(default_factory=HealthMetrics)
    llm_feedback: str = ""
    warnings: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """User profile with preferences."""
    
    default_dataset_types: list[str] = Field(default_factory=list)
    preferred_difficulty: str = "medium"
    preferred_tone: str = "technical"
    default_qa_count: int = 10
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class RunSummary(BaseModel):
    """Summary of a single run."""
    
    run_id: str
    timestamp: datetime
    prompt: str
    dataset_types: list[str]
    qa_per_type: int
    overall_rating: float
    output_path: str


class DatasetConstraints(BaseModel):
    """Advanced constraints for dataset quality control.
    
    These constraints are used by the critic to filter low-quality items
    and ensure the dataset meets production requirements.
    """
    
    # Difficulty distribution target (must sum to 100)
    difficulty_distribution: dict[str, int] = Field(
        default_factory=lambda: {"easy": 30, "medium": 50, "hard": 20},
        description="Target percentage for each difficulty level (must sum to 100)",
    )
    
    # Minimum answer length in characters
    min_answer_length: int = Field(
        default=50,
        ge=0,
        description="Minimum answer length in characters",
    )
    
    # Required code ratio (0-100%) for code LLM datasets
    require_code_ratio: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Minimum percentage of answers that must contain code (0-100)",
    )
    
    # Similarity threshold for duplicate rejection (0.0-1.0)
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold for duplicate detection (0.0-1.0)",
    )
    
    # Banned phrases that critic should flag
    banned_phrases: list[str] = Field(
        default_factory=list,
        description="Comma-separated list of phrases that should be flagged",
    )
    
    @classmethod
    def from_string(cls, banned_str: str) -> "DatasetConstraints":
        """Create constraints with banned phrases from comma-separated string."""
        phrases = [p.strip() for p in banned_str.split(",") if p.strip()]
        return cls(banned_phrases=phrases)


class UserConstraints(BaseModel):
    """User-provided constraints for dataset generation."""
    
    model_config = {"protected_namespaces": ()}
    
    tone: str = "technical"
    difficulty: str = "medium"
    domain: str = ""
    file_paths: list[str] = Field(default_factory=list)
    additional_notes: str = ""
    # V2 additions
    model_family: ModelFamily = ModelFamily.CODE_LLM
    aggressive_filtering: bool = False
    # V2.1 additions: advanced quality constraints
    dataset_constraints: DatasetConstraints = Field(default_factory=DatasetConstraints)


class GenerationRequest(BaseModel):
    """Request for generating datasets."""
    
    prompt: str
    dataset_types: list[str]
    qa_per_type: int
    constraints: UserConstraints
    # V2 additions
    use_llm: bool = True  # Whether to use LLM for generation
    batch_size: int = 10  # Generate in batches to avoid prompt overload


# =============================================================================
# Intent Models (V2)
# =============================================================================

class DatasetIntent(BaseModel):
    """An intent/theme for dataset generation."""
    
    label: str
    description: str
    example_questions: list[str] = Field(default_factory=list)
