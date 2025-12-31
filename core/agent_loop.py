"""
Agentic Loop Orchestrator for the Text2Cypher pipeline.

This module implements the main state machine for the agentic loop:
GENERATE -> VALIDATE -> DECIDE -> (REFINE | EXIT)
"""
import logging
from typing import Optional, Callable, List
from neo4j import Driver

from .agent_state import AgentState, AgentStatus, Attempt, ValidationResult
from .llm_client import LLMClient, build_system_prompt
from config.settings import MAX_ITERATIONS, TEMPERATURE, REFINEMENT_TEMPERATURE

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Main orchestrator for the agentic text2cypher pipeline.

    Implements a state machine with four main states:
    GENERATE -> VALIDATE -> DECIDE -> (REFINE | EXIT)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        driver: Driver,
        validation_pipeline,  # ValidationPipeline type
        feedback_builder,  # FeedbackBuilder type
        base_system_prompt: str,
        schema_content: str,
        max_iterations: int = MAX_ITERATIONS,
        on_iteration_complete: Optional[Callable[[AgentState, Attempt], None]] = None,
    ):
        """
        Initialize the agentic loop.

        Args:
            llm_client: LLM API client
            driver: Neo4j database driver
            validation_pipeline: Orchestrates all validators
            feedback_builder: Builds refinement feedback
            base_system_prompt: Base system prompt with schema
            schema_content: The KG schema content for feedback
            max_iterations: Maximum refinement iterations
            on_iteration_complete: Optional callback after each iteration
        """
        self.llm_client = llm_client
        self.driver = driver
        self.validation_pipeline = validation_pipeline
        self.feedback_builder = feedback_builder
        self.base_system_prompt = base_system_prompt
        self.schema_content = schema_content
        self.max_iterations = max_iterations
        self.on_iteration_complete = on_iteration_complete

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the agentic loop for a single question.

        Args:
            state: Initial agent state with question and config

        Returns:
            Final agent state with results
        """
        logger.info(
            f"Starting agentic loop for question {state.question_id}: "
            f"{state.question[:50]}..."
        )

        while state.status not in [
            AgentStatus.SUCCESS,
            AgentStatus.MAX_ITERATIONS_REACHED,
            AgentStatus.FAILED,
        ]:
            state = self._execute_iteration(state)

        return self._finalize(state)

    def _execute_iteration(self, state: AgentState) -> AgentState:
        """Execute a single iteration of the loop."""

        # Check max iterations
        if state.current_iteration >= self.max_iterations:
            state.status = AgentStatus.MAX_ITERATIONS_REACHED
            logger.info(
                f"Question {state.question_id}: Max iterations reached "
                f"({self.max_iterations})"
            )
            return state

        logger.debug(
            f"Question {state.question_id}: Starting iteration "
            f"{state.current_iteration + 1}/{self.max_iterations}"
        )

        # GENERATE phase
        state.status = AgentStatus.GENERATING
        state = self._generate(state)

        # VALIDATE phase
        state.status = AgentStatus.VALIDATING
        attempt = self._validate(state)
        state.add_attempt(attempt)

        # Callback if provided
        if self.on_iteration_complete:
            self.on_iteration_complete(state, attempt)

        # DECIDE phase
        if attempt.is_valid:
            state.status = AgentStatus.SUCCESS
            logger.info(
                f"Question {state.question_id}: SUCCESS at iteration "
                f"{state.current_iteration + 1}"
            )
        elif state.current_iteration >= self.max_iterations - 1:
            state.status = AgentStatus.MAX_ITERATIONS_REACHED
            logger.info(
                f"Question {state.question_id}: Max iterations reached, "
                f"last error: {attempt.primary_error}"
            )
        else:
            # REFINE phase - prepare for next iteration
            state.status = AgentStatus.REFINING
            logger.debug(
                f"Question {state.question_id}: Refinement needed, "
                f"error: {attempt.primary_error}"
            )

        state.current_iteration += 1
        return state

    def _generate(self, state: AgentState) -> AgentState:
        """Generate a Cypher query using the LLM."""

        if state.current_iteration == 0:
            # First iteration: use base prompt
            system_prompt = self.base_system_prompt
            user_prompt = state.question
            temperature = TEMPERATURE
        else:
            # Refinement iteration: use feedback prompt
            previous_attempt = state.get_previous_attempt()
            feedback = self.feedback_builder.build_feedback(state, previous_attempt)

            system_prompt = self._build_refinement_system_prompt()
            user_prompt = self._build_refinement_user_prompt(
                state.question, feedback
            )
            temperature = REFINEMENT_TEMPERATURE

        # Call LLM
        result = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        state.current_query = result["cypher"]
        state.current_reasoning = result.get("reasoning", "")

        logger.debug(
            f"Question {state.question_id}: Generated query: "
            f"{state.current_query[:100]}..."
        )

        return state

    def _validate(self, state: AgentState) -> Attempt:
        """Validate the current query."""

        all_valid, validation_results = self.validation_pipeline.validate(
            generated_query=state.current_query,
            ground_truth_query=state.ground_truth_query,
        )

        # Build feedback if validation failed and more iterations available
        feedback = None
        if not all_valid and state.current_iteration < self.max_iterations - 1:
            attempt_for_feedback = Attempt(
                iteration=state.current_iteration,
                generated_query=state.current_query,
                reasoning=state.current_reasoning,
                validation_results=validation_results,
                feedback_given=None,
            )
            feedback = self.feedback_builder.build_feedback(state, attempt_for_feedback)

        return Attempt(
            iteration=state.current_iteration,
            generated_query=state.current_query,
            reasoning=state.current_reasoning,
            validation_results=validation_results,
            feedback_given=feedback,
        )

    def _build_refinement_system_prompt(self) -> str:
        """Build the system prompt for refinement iterations."""
        return f"""{self.base_system_prompt}

PENTING: Anda sedang dalam mode koreksi. Kueri sebelumnya gagal validasi.
Baca umpan balik dengan cermat dan perbaiki kesalahan.
Hanya tulis kueri Cypher yang sudah diperbaiki, tanpa penjelasan."""

    def _build_refinement_user_prompt(
        self, original_question: str, feedback: str
    ) -> str:
        """Build the user prompt for refinement iterations."""
        return f"""{feedback}

### Pertanyaan Asli
{original_question}

### Instruksi
Berdasarkan umpan balik di atas, tulis kueri Cypher yang DIPERBAIKI.
Hanya tulis kueri Cypher, tanpa penjelasan.
Tulislah query Cypher dalam 1 baris saja, tanpa adanya baris baru."""

    def _finalize(self, state: AgentState) -> AgentState:
        """Finalize the agent state with results."""

        # Get the last attempt
        final_attempt = state.get_previous_attempt()

        if final_attempt:
            state.final_query = final_attempt.generated_query
            state.final_validation = final_attempt.validation_results
            state.success = final_attempt.is_valid
        else:
            state.success = False

        state.total_iterations = len(state.attempts)

        logger.info(
            f"Question {state.question_id}: Finalized with "
            f"success={state.success}, iterations={state.total_iterations}"
        )

        return state


class BatchAgentRunner:
    """
    Runs the agentic loop on a batch of questions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        driver: Driver,
        validation_pipeline,
        feedback_builder,
        prompt_template_content: str,
        schema_content: str,
        prompt_type: str,
        schema_type: str,
        max_iterations: int = MAX_ITERATIONS,
        rate_limit_delay: float = 3.0,
    ):
        """Initialize batch runner."""
        self.llm_client = llm_client
        self.driver = driver
        self.validation_pipeline = validation_pipeline
        self.feedback_builder = feedback_builder
        self.prompt_type = prompt_type
        self.schema_type = schema_type
        self.max_iterations = max_iterations
        self.rate_limit_delay = rate_limit_delay

        # Build base system prompt
        self.base_system_prompt = build_system_prompt(
            prompt_template_content, schema_content
        )
        self.schema_content = schema_content

    def run_batch(
        self,
        questions: List[dict],
        progress_callback: Optional[Callable[[int, int, AgentState], None]] = None,
    ) -> List[AgentState]:
        """
        Run agentic loop on all questions.

        Args:
            questions: List of question dicts with keys:
                      id, question, ground_truth, metadata
            progress_callback: Optional callback(current, total, state)

        Returns:
            List of final AgentState objects
        """
        import time

        results = []
        total = len(questions)

        # Create agent loop for this batch
        agent_loop = AgentLoop(
            llm_client=self.llm_client,
            driver=self.driver,
            validation_pipeline=self.validation_pipeline,
            feedback_builder=self.feedback_builder,
            base_system_prompt=self.base_system_prompt,
            schema_content=self.schema_content,
            max_iterations=self.max_iterations,
        )

        for i, q in enumerate(questions):
            # Create initial state
            state = AgentState(
                question_id=q["id"],
                question=q["question"],
                ground_truth_query=q["ground_truth"],
                question_metadata=q.get("metadata", {}),
                prompt_type=self.prompt_type,
                schema_type=self.schema_type,
                max_iterations=self.max_iterations,
            )

            # Run agentic loop
            final_state = agent_loop.run(state)
            results.append(final_state)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total, final_state)

            # Rate limiting
            if i < total - 1:  # Don't sleep after last question
                time.sleep(self.rate_limit_delay)

        return results
