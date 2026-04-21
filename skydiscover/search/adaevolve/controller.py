"""
AdaEvolve Controller - Evolution loop with adaptive search intensity.

A clean implementation that uses the adaptive database for all
exploration/exploitation decisions. No explicit stagnation tracking -
search intensity handles exploration automatically.

Features:
- Adaptive sampling based on accumulated improvement signal
- Mode-aware prompting (exploration vs exploitation)
- Paradigm breakthrough for high-level strategy shifts
- Sibling context for learning from previous attempts
- Comprehensive JSON logging of all AdaEvolve signals
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class AdaEvolveController(DiscoveryController):
    """
    AdaEvolve evolution controller with adaptive search intensity.

    Key Features:
    1. Adaptive sampling: Search intensity per island determines exploration/exploitation
    2. Mode-aware prompting: Different guidance for exploration vs exploitation
    3. Sibling context: Shows previous mutations for learning
    4. Error retry: Retries failed generations with error context
    5. Island rotation: UCB-based selection via database.end_iteration()
    6. Paradigm breakthrough: High-level strategy shifts when globally stuck

    No explicit stagnation tracking - search intensity handles exploration
    automatically based on accumulated improvement signal.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self.context_builder = AdaEvolveContextBuilder(self.config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.info(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None

        # Runtime overrides: checked at the start of each iteration.
        # Users can edit {output_dir}/runtime_overrides.yaml during a run
        # and changes take effect on the next iteration.
        self._runtime_overrides_path: Optional[str] = None
        self._runtime_overrides_mtime: float = 0.0

        logger.info(
            f"AdaEvolveController initialized "
            f"(language={self.config.language}, "
            f"paradigm_breakthrough={self.database.use_paradigm_breakthrough})"
        )

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # Runtime config hot-reload
    # =========================================================================

    def _apply_runtime_overrides(self) -> None:
        """Check for runtime_overrides.yaml and apply changes.

        Called at the start of each iteration. Only re-reads the file if it
        has been modified since the last check (stat mtime comparison).

        Supported overrides (edit during a run, takes effect next iteration)::

            # runtime_overrides.yaml
            max_tokens: 16000
            temperature: 0.8
            num_context_programs: 2
            model_weights:                  # rebalance model sampling
              us.anthropic.claude-opus-4-6-v1: 1.0
              us.anthropic.claude-sonnet-4-6: 2.0
        """
        if not self._runtime_overrides_path:
            # Auto-discover: {output_dir}/runtime_overrides.yaml
            out = self.output_dir or getattr(self.database.config, "db_path", None)
            if out:
                self._runtime_overrides_path = os.path.join(out, "runtime_overrides.yaml")
                # Seed an empty file so the user knows where to edit
                if not os.path.exists(self._runtime_overrides_path):
                    try:
                        with open(self._runtime_overrides_path, "w") as f:
                            f.write(
                                "# Runtime overrides — edit during a run, changes apply next iteration.\n"
                                "# Supported keys: max_tokens, temperature, num_context_programs, model_weights\n"
                                "# Example:\n"
                                "#   max_tokens: 16000\n"
                                "#   temperature: 0.8\n"
                                "#   max_parallel_iterations: 5  (requires restart)\n"
                            )
                    except Exception:
                        pass
            else:
                return

        path = self._runtime_overrides_path
        if not os.path.exists(path):
            return

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime <= self._runtime_overrides_mtime:
            return  # unchanged since last read

        self._runtime_overrides_mtime = mtime
        try:
            with open(path) as f:
                overrides = yaml.safe_load(f)
            if not isinstance(overrides, dict) or not overrides:
                return

            applied = []

            # max_tokens — update all LLM clients
            if "max_tokens" in overrides:
                val = int(overrides["max_tokens"])
                for model in self.llms.models:
                    model.max_tokens = val
                applied.append(f"max_tokens={val}")

            # temperature — update all LLM clients
            if "temperature" in overrides:
                val = float(overrides["temperature"])
                for model in self.llms.models:
                    model.temperature = val
                applied.append(f"temperature={val}")

            # num_context_programs
            if "num_context_programs" in overrides:
                val = int(overrides["num_context_programs"])
                self.num_context_programs = val
                applied.append(f"num_context_programs={val}")

            # max_parallel_iterations — can't change the semaphore mid-run,
            # but log it so the user knows to restart
            if "max_parallel_iterations" in overrides:
                val = int(overrides["max_parallel_iterations"])
                current = getattr(self.config, "max_parallel_iterations", 1)
                if val != current:
                    logger.warning(
                        f"max_parallel_iterations changed {current} → {val} in overrides. "
                        f"This requires a restart to take effect."
                    )

            # model_weights — rebalance which model gets sampled
            if "model_weights" in overrides:
                weights_map = overrides["model_weights"]
                if isinstance(weights_map, dict):
                    new_weights = []
                    for cfg in self.llms.models_cfg:
                        w = weights_map.get(cfg.name, cfg.weight)
                        new_weights.append(float(w))
                    total = sum(new_weights) or 1.0
                    self.llms.weights = [w / total for w in new_weights]
                    applied.append(f"model_weights={weights_map}")

            if applied:
                logger.info(f"Runtime overrides applied: {', '.join(applied)}")

        except Exception as e:
            logger.warning(f"Failed to apply runtime overrides from {path}: {e}")

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )

        logger.info(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
            }

            # Add child program info if available
            if child_program:
                metadata = child_program.get("metadata", {})
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                    "changes": metadata.get("changes", ""),
                    "island_id": metadata.get("island_id"),
                    "sampling_mode": metadata.get("sampling_mode"),
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation.

        When ``config.max_parallel_iterations > 1``, iterations run
        concurrently as asyncio tasks bounded by a semaphore.  While
        iteration *i* evaluates (compile + sim), iterations *i+1..i+K*
        can generate LLM solutions and start their own evals, keeping
        build and sim workers saturated.
        """
        import asyncio

        total = start_iteration + max_iterations
        max_parallel = getattr(self.config, "max_parallel_iterations", 1)
        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands "
            f"(max_parallel={max_parallel})"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        if max_parallel <= 1:
            # Sequential path — original behaviour
            for iteration in range(start_iteration, total):
                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested")
                    break
                self._apply_runtime_overrides()
                try:
                    await self._run_iteration(iteration, checkpoint_callback)
                except Exception as e:
                    logger.exception(f"Iteration {iteration} failed: {e}")
                finally:
                    self.database.end_iteration(iteration)
        else:
            # Parallel path — up to K iterations in flight
            sem = asyncio.Semaphore(max_parallel)
            # Lock for database mutations (_process_result + end_iteration)
            db_lock = asyncio.Lock()
            pending: set = set()

            async def _bounded_iteration(iteration: int) -> int:
                # The semaphore gates how many iterations can START (LLM gen +
                # parent sampling) at once.  It's released as soon as the
                # iteration enters evaluation (compile+sim dispatched to
                # workers), so new iterations can begin their LLM phase while
                # previous iterations' long sims (~20 min) are running.  This
                # keeps build workers continuously busy.
                #
                # The semaphore is threaded through _run_normal_step and
                # _generate_child via self._iteration_sem so it can be released
                # at the right point (after LLM gen, before eval await).
                await sem.acquire()
                if self.shutdown_event.is_set():
                    sem.release()
                    return iteration
                self._apply_runtime_overrides()
                if (self.database.use_paradigm_breakthrough
                        and self.database.is_paradigm_stagnating()):
                    await self._generate_paradigms_if_needed()

                iteration_start_time = time.time()
                try:
                    result = await self._run_normal_step(iteration, release_sem=sem)
                except Exception as e:
                    logger.exception(f"Iteration {iteration} failed: {e}")
                    async with db_lock:
                        self.database.end_iteration(iteration)
                    return iteration

                iteration_time = time.time() - iteration_start_time

                # Process result under lock (database.add, monitor callback,
                # end_iteration are all sync and must not interleave)
                async with db_lock:
                    if result.error:
                        logger.warning(f"Iteration {iteration}: {result.error}")
                        self._log_iteration_stats(
                            iteration=iteration,
                            sampling_mode=self._last_sampling_mode,
                            sampling_intensity=self._last_sampling_intensity,
                            child_program=None,
                            iteration_time=iteration_time,
                            llm_generation_time=result.llm_generation_time,
                            eval_time=result.eval_time,
                            error=result.error,
                        )
                    else:
                        self._process_result(result, iteration, checkpoint_callback)
                        self._log_iteration_stats(
                            iteration=iteration,
                            sampling_mode=self._last_sampling_mode,
                            sampling_intensity=self._last_sampling_intensity,
                            child_program=result.child_program_dict,
                            iteration_time=result.iteration_time,
                            llm_generation_time=result.llm_generation_time,
                            eval_time=result.eval_time,
                            error=None,
                        )
                    self.database.end_iteration(iteration)

                return iteration

            for iteration in range(start_iteration, total):
                if self.shutdown_event.is_set():
                    break
                task = asyncio.create_task(
                    _bounded_iteration(iteration),
                    name=f"adaevolve_iter_{iteration}",
                )
                pending.add(task)
                task.add_done_callback(pending.discard)

                # Backpressure: when pipeline is full, wait for one to finish
                if len(pending) >= max_parallel:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in done:
                        try:
                            t.result()  # propagate exceptions
                        except Exception as e:
                            logger.warning(f"Parallel iteration exception: {e}")

            # Drain remaining
            if pending:
                done, _ = await asyncio.wait(pending)
                for t in done:
                    try:
                        t.result()
                    except Exception as e:
                        logger.warning(f"Parallel iteration exception: {e}")

        logger.info("AdaEvolve completed")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.info(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")

        return self.database.get_best_program()

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program."""
        # Find a seed program
        seed_program = None
        for i in range(self.database.num_islands):
            size = self.database.get_island_size(i)
            if size > 0 and seed_program is None:
                population = self.database.get_island_population(i)
                if population:
                    seed_program = population[0]
                    break

        if seed_program is None:
            logger.warning("No seed program found")
            return

        # Seed empty islands
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed_program.solution,
                    language=seed_program.language,
                    metrics=seed_program.metrics.copy() if seed_program.metrics else {},
                    iteration_found=seed_program.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.info(f"Seeded island {i}")

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        """Execute one evolution iteration."""
        iteration_start_time = time.time()

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            await self._generate_paradigms_if_needed()

        result = await self._run_normal_step(iteration)

        iteration_time = time.time() - iteration_start_time

        if result.error:
            logger.warning(f"Iteration {iteration}: {result.error}")
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
        else:
            self._process_result(result, iteration, checkpoint_callback)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )

    async def _generate_paradigms_if_needed(self) -> None:
        """Generate new paradigms if stagnating and none active."""
        if self.paradigm_generator is None:
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.info(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(self, iteration: int, release_sem=None) -> SerializableResult:
        """Run a normal iteration with optional retry.

        Args:
            release_sem: If provided, an asyncio.Semaphore to release after
                the LLM generates code and evaluation is dispatched.  This
                lets the next iteration start its LLM phase while this one's
                sims are still running.
        """
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            result = await self._generate_child(
                iteration, error_context=last_error, release_sem=release_sem,
            )
            release_sem = None  # only release once (first attempt)
            if not result.error:
                return result
            last_error = result.error
            logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
        )

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # Add to database (database handles which island)
        self.database.add(child, iteration=iteration, parent_id=result.parent_id)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        changes = child.metadata.get("changes", "")
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )
        if changes:
            logger.info(f"  Changes: {changes}")

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.info(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

        # Checkpoint callback
        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        release_sem=None,
    ) -> SerializableResult:
        """Generate and evaluate a single child program."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            # Sample parent and context programs (database returns standard framework dicts)
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
            )

            # Unpack parent dict (standard framework pattern)
            if not parent_dict:
                logger.error("sample() returned empty parent dict")
                return SerializableResult(
                    error="Empty parent dict from sample()", iteration=iteration
                )
            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]

            # Read sampling mode stashed by database.sample()
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"

            # Capture sampling mode and intensity for logging
            self._last_sampling_mode = sampling_mode
            current_island = self.database.current_island
            if self.database.use_adaptive_search:
                self._last_sampling_intensity = self.database.adapter.get_search_intensity(
                    current_island
                )
            else:
                self._last_sampling_intensity = self.database.fixed_intensity

            # When paradigm is active, use best program as parent
            # This ensures paradigm (designed from best) is applied to best, not random parent
            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            if paradigm:
                best_program = self.database.get_best_program()
                if best_program:
                    parent_dict = {parent_label: best_program}
                    parent = best_program
                    # Keep context_programs_dict from sampling for diversity

            # Gather siblings for sibling context
            siblings = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                # AdaEvolve-specific keys (consumed by AdaEvolveContextBuilder)
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
            }
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (AdaEvolveContextBuilder handles paradigm/sibling/error formatting)
            prompt = self.context_builder.build_prompt(parent_dict, context)

            # Mark paradigm as used after prompt is built
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Generate and evaluate
            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
                release_sem=release_sem,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            if release_sem is not None:
                try:
                    release_sem.release()
                except ValueError:
                    pass
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
        release_sem=None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from skydiscover.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                )
                response = result.text or ""
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image", iteration=iteration
                    )
            else:
                result = await self._call_llm(prompt["system"], prompt["user"])
                response = result.text
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            return SerializableResult(error=f"LLM error: {e}", iteration=iteration)

        if not response and self.config.language != "image":
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        # Release the iteration semaphore before the long eval await.
        # LLM generation is done, code is parsed — the next iteration can now
        # start its LLM call while this one compiles + simulates on workers.
        if release_sem is not None:
            try:
                release_sem.release()
            except ValueError:
                pass  # already released

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            return SerializableResult(error=f"Evaluation error: {e}", iteration=iteration)

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {"changes": changes, "parent_metrics": parent.metrics}
        if image_path:
            child_metadata["image_path"] = image_path
        # Capture LLM call metadata (model used, token counts) — set by openai.py
        try:
            if hasattr(result, "model_used") and result.model_used:
                child_metadata["llm_model_used"] = result.model_used
            if hasattr(result, "prompt_tokens") and result.prompt_tokens is not None:
                child_metadata["llm_prompt_tokens"] = result.prompt_tokens
            if hasattr(result, "completion_tokens") and result.completion_tokens is not None:
                child_metadata["llm_completion_tokens"] = result.completion_tokens
            if hasattr(result, "total_tokens") and result.total_tokens is not None:
                child_metadata["llm_total_tokens"] = result.total_tokens
            child_metadata["llm_generation_time_sec"] = llm_generation_time
        except Exception:
            pass
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
        )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            iteration=iteration,
        )
