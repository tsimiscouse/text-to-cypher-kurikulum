"""
Batch Processor for the Agentic Text2Cypher pipeline (Ollama Version).

Uses local Ollama LLM instead of cloud API.
"""
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase

from config.settings import Settings
from config.llm_config_ollama import LLMConfigOllama
from prompts.prompt_manager import PromptManager, PromptType, SchemaFormat
from data.ground_truth_loader import GroundTruthItem
from core.llm_client import LLMClient
from core.agent_loop import AgentLoop
from core.agent_state import AgentState
from validators.validation_pipeline import ValidationPipeline
from feedback.feedback_builder import FeedbackBuilder

logger = logging.getLogger(__name__)


@dataclass
class BatchProgress:
    """Tracks batch processing progress."""

    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    start_time: Optional[str] = None
    last_processed_id: Optional[int] = None
    processed_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "start_time": self.start_time,
            "last_processed_id": self.last_processed_id,
            "processed_ids": self.processed_ids,
            "completion_percentage": (
                (self.processed_items / self.total_items * 100)
                if self.total_items > 0 else 0
            )
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProgress":
        """Create from dictionary."""
        return cls(
            total_items=data.get("total_items", 0),
            processed_items=data.get("processed_items", 0),
            successful_items=data.get("successful_items", 0),
            failed_items=data.get("failed_items", 0),
            start_time=data.get("start_time"),
            last_processed_id=data.get("last_processed_id"),
            processed_ids=data.get("processed_ids", [])
        )


class BatchProcessorOllama:
    """
    Processes questions in batches using Ollama (Local LLM).

    Features:
    - No rate limiting (local inference)
    - Progress saving for resumability
    - Detailed logging and error handling
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_config: Optional[LLMConfigOllama] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize the batch processor for Ollama.

        Args:
            settings: Application settings
            llm_config: Ollama LLM configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        self.settings = settings or Settings()
        self.llm_config = llm_config or LLMConfigOllama()

        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints_ollama"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.prompt_manager = PromptManager()
        self.llm_client = LLMClient(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            model=self.llm_config.model,
        )

        # Neo4j driver (lazy initialization)
        self._neo4j_driver = None

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

    def process_batch(
        self,
        items: List[GroundTruthItem],
        prompt_type: PromptType,
        schema_format: SchemaFormat,
        batch_id: str,
        on_item_complete: Optional[Callable[[int, AgentState], None]] = None,
        resume: bool = True
    ) -> List[AgentState]:
        """
        Process a batch of questions.

        Args:
            items: List of ground truth items to process
            prompt_type: Prompt engineering technique to use
            schema_format: Schema representation format to use
            batch_id: Unique identifier for this batch
            on_item_complete: Callback when an item is completed
            resume: Whether to resume from checkpoint if available

        Returns:
            List of AgentState objects for all processed items
        """
        config_name = self.prompt_manager.get_configuration_name(prompt_type, schema_format)
        checkpoint_path = self.checkpoint_dir / f"{batch_id}_checkpoint.json"

        # Load or initialize progress
        progress = self._load_checkpoint(checkpoint_path) if resume else None
        if progress is None:
            progress = BatchProgress(
                total_items=len(items),
                start_time=datetime.now().isoformat()
            )

        # Initialize components
        schema_content = self.prompt_manager.load_schema(schema_format)
        validation_pipeline = ValidationPipeline(
            driver=self.neo4j_driver
        )
        feedback_builder = FeedbackBuilder(schema_content)

        agent_loop = AgentLoop(
            llm_client=self.llm_client,
            validation_pipeline=validation_pipeline,
            feedback_builder=feedback_builder,
            prompt_manager=self.prompt_manager,
            prompt_type=prompt_type,
            schema_format=schema_format,
            max_iterations=self.settings.max_iterations
        )

        # Process items
        results: List[AgentState] = []
        items_to_process = [
            item for item in items
            if item.id not in progress.processed_ids
        ]

        logger.info(
            f"[Ollama] Starting batch {batch_id} for {config_name}: "
            f"{len(items_to_process)} items to process "
            f"({progress.processed_items}/{progress.total_items} already done)"
        )

        for i, item in enumerate(items_to_process):
            try:
                logger.info(
                    f"[{progress.processed_items + 1}/{progress.total_items}] "
                    f"Processing Q{item.id}: {item.question[:40]}..."
                )

                # Process the question
                state = agent_loop.run(
                    question_id=item.id,
                    question=item.question,
                    ground_truth_query=item.cypher_query
                )
                results.append(state)

                # Update progress
                progress.processed_items += 1
                progress.last_processed_id = item.id
                progress.processed_ids.append(item.id)

                if state.success:
                    progress.successful_items += 1
                else:
                    progress.failed_items += 1

                # Callback
                if on_item_complete:
                    on_item_complete(item.id, state)

                # Save checkpoint
                self._save_checkpoint(checkpoint_path, progress, results)

                # No rate limiting needed for local LLM!

            except Exception as e:
                logger.error(f"Error processing question {item.id}: {e}")
                progress.failed_items += 1
                progress.processed_items += 1
                progress.processed_ids.append(item.id)

                # Save checkpoint even on error
                self._save_checkpoint(checkpoint_path, progress, results)

        logger.info(
            f"[Ollama] Batch {batch_id} complete: "
            f"{progress.successful_items} successful, "
            f"{progress.failed_items} failed"
        )

        return results

    def _load_checkpoint(self, checkpoint_path: Path) -> Optional[BatchProgress]:
        """Load progress from checkpoint file."""
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return BatchProgress.from_dict(data.get("progress", {}))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        progress: BatchProgress,
        results: List[AgentState]
    ):
        """Save progress to checkpoint file."""
        try:
            checkpoint_data = {
                "progress": progress.to_dict(),
                "last_updated": datetime.now().isoformat(),
                "results_summary": [
                    {
                        "question_id": state.question_id,
                        "success": state.success,
                        "iterations": state.total_iterations,
                        "final_query": state.final_query
                    }
                    for state in results
                ]
            }

            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def process_single(
        self,
        item: GroundTruthItem,
        prompt_type: PromptType,
        schema_format: SchemaFormat
    ) -> AgentState:
        """
        Process a single question.

        Args:
            item: The ground truth item to process
            prompt_type: Prompt engineering technique to use
            schema_format: Schema representation format to use

        Returns:
            AgentState for the processed item
        """
        schema_content = self.prompt_manager.load_schema(schema_format)
        prompt_template = self.prompt_manager.load_prompt_template(prompt_type)

        # Build the base system prompt
        from core.llm_client import build_system_prompt
        base_system_prompt = build_system_prompt(prompt_template, schema_content)

        validation_pipeline = ValidationPipeline(
            driver=self.neo4j_driver
        )
        feedback_builder = FeedbackBuilder(schema_content)

        agent_loop = AgentLoop(
            llm_client=self.llm_client,
            driver=self.neo4j_driver,
            validation_pipeline=validation_pipeline,
            feedback_builder=feedback_builder,
            base_system_prompt=base_system_prompt,
            schema_content=schema_content,
            max_iterations=self.settings.max_iterations
        )

        # Create initial state
        state = AgentState(
            question_id=item.id,
            question=item.question,
            ground_truth_query=item.cypher_query,
            prompt_type=prompt_type.value,
            schema_type=schema_format.value,
            max_iterations=self.settings.max_iterations
        )

        return agent_loop.run(state)


def create_batch_processor_ollama(
    settings: Optional[Settings] = None,
    llm_config: Optional[LLMConfigOllama] = None,
    checkpoint_dir: Optional[str] = None,
) -> BatchProcessorOllama:
    """
    Factory function to create a BatchProcessorOllama.

    Args:
        settings: Application settings
        llm_config: Ollama LLM configuration
        checkpoint_dir: Directory for checkpoints

    Returns:
        Configured BatchProcessorOllama instance
    """
    return BatchProcessorOllama(
        settings=settings,
        llm_config=llm_config,
        checkpoint_dir=checkpoint_dir,
    )
