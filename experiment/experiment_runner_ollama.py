"""
Experiment Runner for the Agentic Text2Cypher pipeline (Ollama Version).

Uses local Ollama LLM instead of cloud API.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import csv

from neo4j import GraphDatabase

from config.settings import Settings
from config.llm_config_ollama import LLMConfigOllama
from prompts.prompt_manager import PromptManager, PromptType, SchemaFormat
from data.ground_truth_loader import GroundTruthLoader, GroundTruthItem
from core.llm_client import LLMClient
from core.agent_loop import AgentLoop
from core.agent_state import AgentState
from validators.validation_pipeline import ValidationPipeline
from feedback.feedback_builder import FeedbackBuilder
from metrics import (
    calculate_all_cypher_metrics,
    calculate_output_metrics,
    calculate_agentic_metrics,
    calculate_llmetric_q,
    calculate_llmetric,
    AgenticMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Result for a single question."""

    question_id: int
    question: str
    ground_truth_query: str
    generated_query: str
    success: bool
    total_iterations: int
    first_attempt_valid: bool

    # Cypher metrics
    bleu: float = 0.0
    rouge_l_f1: float = 0.0
    jaro_winkler: float = 0.0
    jaccard_cypher: float = 0.0

    # Output metrics
    pass_at_1: bool = False
    jaccard_output: float = 0.0

    # LLMetric-Q
    llmetric_q: float = 0.0

    # Metadata
    reasoning_level: str = ""
    sublevel: str = ""
    complexity: str = ""
    errors_encountered: List[str] = field(default_factory=list)
    iteration_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConfigurationResult:
    """Result for a single configuration (prompt × schema combination)."""

    config_name: str
    prompt_type: str
    schema_format: str
    question_results: List[QuestionResult] = field(default_factory=list)
    agentic_metrics: Optional[AgenticMetrics] = None

    # Aggregated metrics
    pass_at_1_rate: float = 0.0
    kg_valid_rate: float = 0.0
    avg_bleu: float = 0.0
    avg_rouge_l_f1: float = 0.0
    avg_jaro_winkler: float = 0.0
    avg_jaccard_cypher: float = 0.0
    avg_jaccard_output: float = 0.0
    avg_llmetric_q: float = 0.0
    llmetric: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_name": self.config_name,
            "prompt_type": self.prompt_type,
            "schema_format": self.schema_format,
            "pass_at_1_rate": self.pass_at_1_rate,
            "kg_valid_rate": self.kg_valid_rate,
            "avg_bleu": self.avg_bleu,
            "avg_rouge_l_f1": self.avg_rouge_l_f1,
            "avg_jaro_winkler": self.avg_jaro_winkler,
            "avg_jaccard_cypher": self.avg_jaccard_cypher,
            "avg_jaccard_output": self.avg_jaccard_output,
            "avg_llmetric_q": self.avg_llmetric_q,
            "llmetric": self.llmetric,
            "agentic_metrics": self.agentic_metrics.to_dict() if self.agentic_metrics else None,
            "total_questions": len(self.question_results),
        }


class ExperimentRunnerOllama:
    """
    Runs the full experiment across all configurations using Ollama.

    No rate limiting - can run full experiment without delays!
    """

    # All configurations to run
    CONFIGURATIONS = [
        (PromptType.ZERO_SHOT, SchemaFormat.FULL_SCHEMA),
        (PromptType.ZERO_SHOT, SchemaFormat.NODES_PATHS),
        (PromptType.ZERO_SHOT, SchemaFormat.ONLY_PATHS),
        (PromptType.FEW_SHOT, SchemaFormat.FULL_SCHEMA),
        (PromptType.FEW_SHOT, SchemaFormat.NODES_PATHS),
        (PromptType.FEW_SHOT, SchemaFormat.ONLY_PATHS),
        (PromptType.CHAIN_OF_THOUGHT, SchemaFormat.FULL_SCHEMA),
        (PromptType.CHAIN_OF_THOUGHT, SchemaFormat.NODES_PATHS),
        (PromptType.CHAIN_OF_THOUGHT, SchemaFormat.ONLY_PATHS),
    ]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_config: Optional[LLMConfigOllama] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the experiment runner for Ollama.

        Args:
            settings: Application settings
            llm_config: Ollama LLM configuration
            results_dir: Directory to save results
        """
        self.settings = settings or Settings()
        self.llm_config = llm_config or LLMConfigOllama()

        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # Save to separate directory for Ollama results
            self.results_dir = Path(__file__).parent.parent / "results_ollama"

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.prompt_manager = PromptManager()
        self.ground_truth_loader = GroundTruthLoader()
        self.llm_client = LLMClient(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            model=self.llm_config.model,
        )

        # Neo4j driver (lazy initialization)
        self._neo4j_driver = None

        # Results storage
        self.results: Dict[str, ConfigurationResult] = {}

    @property
    def neo4j_driver(self):
        """Get Neo4j driver (lazy initialization)."""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
        return self._neo4j_driver

    def close(self):
        """Close connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()

    def run_all_configurations(
        self,
        max_questions: Optional[int] = None,
        configurations: Optional[List[tuple]] = None
    ) -> Dict[str, ConfigurationResult]:
        """
        Run all configurations.

        Args:
            max_questions: Limit number of questions per config (for testing)
            configurations: Specific configurations to run (default: all 9)

        Returns:
            Dictionary of configuration results
        """
        configs_to_run = configurations or self.CONFIGURATIONS
        ground_truth_items = self.ground_truth_loader.load()

        if max_questions:
            ground_truth_items = ground_truth_items[:max_questions]

        logger.info(
            f"[Ollama] Running {len(configs_to_run)} configurations "
            f"with {len(ground_truth_items)} questions each"
        )

        for prompt_type, schema_format in configs_to_run:
            config_name = self.prompt_manager.get_configuration_name(prompt_type, schema_format)
            logger.info(f"[Ollama] Starting configuration: {config_name}")

            try:
                result = self.run_configuration(
                    prompt_type,
                    schema_format,
                    ground_truth_items
                )
                self.results[config_name] = result

                # Save intermediate results
                self._save_configuration_results(config_name, result)

            except Exception as e:
                logger.error(f"Error running configuration {config_name}: {e}")
                continue

        # Save final summary
        self._save_experiment_summary()

        return self.results

    def run_configuration(
        self,
        prompt_type: PromptType,
        schema_format: SchemaFormat,
        ground_truth_items: List[GroundTruthItem]
    ) -> ConfigurationResult:
        """
        Run a single configuration.

        Args:
            prompt_type: The prompt engineering technique
            schema_format: The schema representation format
            ground_truth_items: List of questions to process

        Returns:
            ConfigurationResult for this configuration
        """
        config_name = self.prompt_manager.get_configuration_name(prompt_type, schema_format)
        schema_content = self.prompt_manager.load_schema(schema_format)
        prompt_template = self.prompt_manager.load_prompt_template(prompt_type)

        # Build the base system prompt
        from core.llm_client import build_system_prompt
        base_system_prompt = build_system_prompt(prompt_template, schema_content)

        # Initialize components for this configuration
        validation_pipeline = ValidationPipeline(
            driver=self.neo4j_driver
        )
        feedback_builder = FeedbackBuilder(schema_content)

        # Create agent loop
        agent_loop = AgentLoop(
            llm_client=self.llm_client,
            driver=self.neo4j_driver,
            validation_pipeline=validation_pipeline,
            feedback_builder=feedback_builder,
            base_system_prompt=base_system_prompt,
            schema_content=schema_content,
            max_iterations=self.settings.max_iterations
        )

        # Process all questions
        question_results: List[QuestionResult] = []
        agent_states: List[AgentState] = []

        for item in ground_truth_items:
            logger.info(f"[Ollama] Processing question {item.id}: {item.question[:50]}...")

            try:
                # Create initial state for each question
                initial_state = AgentState(
                    question_id=item.id,
                    question=item.question,
                    ground_truth_query=item.cypher_query,
                    prompt_type=prompt_type.value,
                    schema_type=schema_format.value,
                    max_iterations=self.settings.max_iterations
                )
                state = agent_loop.run(initial_state)
                agent_states.append(state)

                # Calculate metrics for this question
                question_result = self._calculate_question_metrics(item, state)
                question_results.append(question_result)

            except Exception as e:
                logger.error(f"Error processing question {item.id}: {e}")
                # Add failed result
                question_results.append(QuestionResult(
                    question_id=item.id,
                    question=item.question,
                    ground_truth_query=item.cypher_query,
                    generated_query="",
                    success=False,
                    total_iterations=0,
                    first_attempt_valid=False,
                    reasoning_level=item.reasoning_level,
                    sublevel=item.sublevel,
                    complexity=item.complexity,
                    errors_encountered=[str(e)]
                ))

        # Calculate aggregated metrics
        result = self._aggregate_results(
            config_name,
            prompt_type,
            schema_format,
            question_results,
            agent_states
        )

        return result

    def _calculate_question_metrics(
        self,
        item: GroundTruthItem,
        state: AgentState
    ) -> QuestionResult:
        """Calculate all metrics for a single question."""
        generated_query = state.final_query or ""

        # Cypher string metrics
        cypher_metrics = calculate_all_cypher_metrics(
            item.cypher_query,
            generated_query
        )

        # Output metrics (requires Neo4j execution)
        try:
            output_metrics = calculate_output_metrics(
                self.neo4j_driver,
                item.cypher_query,
                generated_query
            )
        except Exception as e:
            logger.warning(f"Failed to calculate output metrics for Q{item.id}: {e}")
            output_metrics = {"pass_at_1": False, "jaccard_output": 0.0}

        # Calculate LLMetric-Q
        llmetric_q = calculate_llmetric_q(
            pass_at_1=output_metrics.get("pass_at_1", False),
            kg_valid=state.success,
            jaccard_output=output_metrics.get("jaccard_output", 0.0),
            jaro_winkler=cypher_metrics.get("jaro_winkler", 0.0),
            rouge_l_f1=cypher_metrics.get("rouge_l_f1", 0.0)
        )

        # Build iteration history
        iteration_history = []
        for attempt in state.attempts:
            iteration_history.append({
                "iteration": attempt.iteration,
                "query": attempt.generated_query,
                "is_valid": attempt.is_valid,
                "errors": [v.error_message for v in attempt.validation_results if not v.is_valid]
            })

        # Collect errors encountered
        errors = []
        for attempt in state.attempts:
            for v in attempt.validation_results:
                if not v.is_valid and v.error_type:
                    errors.append(v.error_type.value)

        return QuestionResult(
            question_id=item.id,
            question=item.question,
            ground_truth_query=item.cypher_query,
            generated_query=generated_query,
            success=state.success,
            total_iterations=state.total_iterations,
            first_attempt_valid=state.attempts[0].is_valid if state.attempts else False,
            bleu=cypher_metrics.get("bleu", 0.0),
            rouge_l_f1=cypher_metrics.get("rouge_l_f1", 0.0),
            jaro_winkler=cypher_metrics.get("jaro_winkler", 0.0),
            jaccard_cypher=cypher_metrics.get("jaccard_cypher", 0.0),
            pass_at_1=output_metrics.get("pass_at_1", False),
            jaccard_output=output_metrics.get("jaccard_output", 0.0),
            llmetric_q=llmetric_q,
            reasoning_level=item.reasoning_level,
            sublevel=item.sublevel,
            complexity=item.complexity,
            errors_encountered=list(set(errors)),
            iteration_history=iteration_history
        )

    def _aggregate_results(
        self,
        config_name: str,
        prompt_type: PromptType,
        schema_format: SchemaFormat,
        question_results: List[QuestionResult],
        agent_states: List[AgentState]
    ) -> ConfigurationResult:
        """Aggregate metrics across all questions."""
        n = len(question_results)
        if n == 0:
            return ConfigurationResult(
                config_name=config_name,
                prompt_type=prompt_type.value,
                schema_format=schema_format.value
            )

        # Calculate agentic metrics
        agentic_metrics = calculate_agentic_metrics(agent_states)

        # Aggregate averages
        pass_at_1_count = sum(1 for r in question_results if r.pass_at_1)
        kg_valid_count = sum(1 for r in question_results if r.success)

        avg_bleu = sum(r.bleu for r in question_results) / n
        avg_rouge_l_f1 = sum(r.rouge_l_f1 for r in question_results) / n
        avg_jaro_winkler = sum(r.jaro_winkler for r in question_results) / n
        avg_jaccard_cypher = sum(r.jaccard_cypher for r in question_results) / n
        avg_jaccard_output = sum(r.jaccard_output for r in question_results) / n
        avg_llmetric_q = sum(r.llmetric_q for r in question_results) / n

        pass_at_1_rate = (pass_at_1_count / n) * 100
        kg_valid_rate = (kg_valid_count / n) * 100

        # Calculate JaRou (average of Jaro-Winkler and Rouge-L F1)
        jarou_avg = ((avg_jaro_winkler + avg_rouge_l_f1) / 2) * 100

        # Calculate LLMetric
        llmetric = calculate_llmetric(
            pass_at_1_rate=pass_at_1_rate,
            kg_valid_rate=kg_valid_rate,
            jaccard_output_avg=avg_jaccard_output * 100,
            jarou_avg=jarou_avg
        )

        return ConfigurationResult(
            config_name=config_name,
            prompt_type=prompt_type.value,
            schema_format=schema_format.value,
            question_results=question_results,
            agentic_metrics=agentic_metrics,
            pass_at_1_rate=pass_at_1_rate,
            kg_valid_rate=kg_valid_rate,
            avg_bleu=avg_bleu,
            avg_rouge_l_f1=avg_rouge_l_f1,
            avg_jaro_winkler=avg_jaro_winkler,
            avg_jaccard_cypher=avg_jaccard_cypher,
            avg_jaccard_output=avg_jaccard_output,
            avg_llmetric_q=avg_llmetric_q,
            llmetric=llmetric
        )

    def _save_configuration_results(self, config_name: str, result: ConfigurationResult):
        """Save results for a single configuration."""
        config_dir = self.results_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results as CSV
        csv_path = config_dir / "agentic_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if result.question_results:
                fieldnames = [
                    "question_id", "question", "ground_truth_query", "generated_query",
                    "success", "total_iterations", "first_attempt_valid",
                    "bleu", "rouge_l_f1", "jaro_winkler", "jaccard_cypher",
                    "pass_at_1", "jaccard_output", "llmetric_q",
                    "reasoning_level", "sublevel", "complexity", "errors_encountered"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for qr in result.question_results:
                    row = {
                        "question_id": qr.question_id,
                        "question": qr.question,
                        "ground_truth_query": qr.ground_truth_query,
                        "generated_query": qr.generated_query,
                        "success": qr.success,
                        "total_iterations": qr.total_iterations,
                        "first_attempt_valid": qr.first_attempt_valid,
                        "bleu": qr.bleu,
                        "rouge_l_f1": qr.rouge_l_f1,
                        "jaro_winkler": qr.jaro_winkler,
                        "jaccard_cypher": qr.jaccard_cypher,
                        "pass_at_1": qr.pass_at_1,
                        "jaccard_output": qr.jaccard_output,
                        "llmetric_q": qr.llmetric_q,
                        "reasoning_level": qr.reasoning_level,
                        "sublevel": qr.sublevel,
                        "complexity": qr.complexity,
                        "errors_encountered": ",".join(qr.errors_encountered)
                    }
                    writer.writerow(row)

        # Save summary as JSON
        summary_path = config_dir / "agentic_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"[Ollama] Saved results for {config_name} to {config_dir}")

    def _save_experiment_summary(self):
        """Save overall experiment summary."""
        summary_path = self.results_dir / "experiment_summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "llm_provider": "ollama",
            "llm_model": self.llm_config.model,
            "total_configurations": len(self.results),
            "configurations": {
                name: result.to_dict()
                for name, result in self.results.items()
            }
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"[Ollama] Saved experiment summary to {summary_path}")


def run_experiment_ollama(
    max_questions: Optional[int] = None,
    configurations: Optional[List[tuple]] = None,
    results_dir: Optional[str] = None
) -> Dict[str, ConfigurationResult]:
    """
    Convenience function to run the experiment with Ollama.

    Args:
        max_questions: Limit questions per config (for testing)
        configurations: Specific configurations to run
        results_dir: Directory to save results

    Returns:
        Dictionary of configuration results
    """
    runner = ExperimentRunnerOllama(results_dir=results_dir)
    try:
        return runner.run_all_configurations(max_questions, configurations)
    finally:
        runner.close()
