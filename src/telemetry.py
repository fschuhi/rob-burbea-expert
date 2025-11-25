"""
Telemetry utilities for tracking RAG query performance.

This module provides tools for:
- Timing different phases of query execution
- Calculating derived metrics (speed, throughput)
- Exporting metrics for display or logging
"""

import time
from typing import Optional


class QueryTelemetry:
    """
    Track timing and performance metrics for RAG queries.

    Supports flexible phase naming and automatic duration calculation.
    Designed for both streaming (Answer Generator) and batch (Search Explorer) queries.

    Example:
        >>> telemetry = QueryTelemetry()
        >>> telemetry.start_query()
        >>>
        >>> telemetry.start_phase("retrieval")
        >>> # ... do retrieval work ...
        >>> telemetry.end_phase("retrieval", metadata={"reranker": True})
        >>>
        >>> telemetry.start_phase("generation")
        >>> # ... generate response ...
        >>> telemetry.end_phase("generation")
        >>> telemetry.calculate_generation_metrics("The generated text...")
        >>>
        >>> metrics = telemetry.to_dict()
        >>> print(metrics)
        {'retrieval': 0.52, 'generation': 7.3, 'speed': 15.2, ...}
    """

    def __init__(self):
        """Initialize telemetry tracker."""
        self.query_start_time: Optional[float] = None
        self.phases: dict[str, dict] = {}
        self.metadata: dict[str, any] = {}
        self._current_phase: Optional[str] = None
        self._phase_start_time: Optional[float] = None

    def start_query(self):
        """Mark the start of query processing."""
        self.query_start_time = time.time()

    def start_phase(self, phase_name: str):
        """
        Mark the start of a named phase.

        Args:
            phase_name: Name of the phase (e.g., "retrieval", "generation")
        """
        if self.query_start_time is None:
            # Auto-start if forgotten
            self.start_query()

        self._current_phase = phase_name
        self._phase_start_time = time.time()

    def end_phase(self, phase_name: Optional[str] = None, metadata: Optional[dict] = None):
        """
        Mark the end of a phase and calculate duration.

        Args:
            phase_name: Name of the phase (optional if using current phase)
            metadata: Optional metadata to attach to this phase
        """
        if phase_name is None:
            phase_name = self._current_phase

        if phase_name is None:
            raise ValueError("No active phase to end. Call start_phase() first.")

        if self._phase_start_time is None:
            raise ValueError(f"Phase '{phase_name}' was never started.")

        end_time = time.time()
        duration = end_time - self._phase_start_time

        self.phases[phase_name] = {"duration": duration, "start": self._phase_start_time, "end": end_time}

        if metadata:
            self.phases[phase_name]["metadata"] = metadata

        # Clear current phase tracking
        self._current_phase = None
        self._phase_start_time = None

    def mark_ttft(self):
        """
        Mark time-to-first-token.

        Convenience method that ends the current phase and stores as "ttft".
        Typically called after first chunk arrives from streaming LLM.
        """
        if self._current_phase:
            self.end_phase("ttft")

    def calculate_generation_metrics(self, generated_text: str):
        """
        Calculate generation speed metrics from the generated text.

        Args:
            generated_text: The text that was generated

        Note:
            Requires that a "generation" phase exists in self.phases.
        """
        if "generation" not in self.phases:
            # If no explicit generation phase, calculate from TTFT to now
            if "ttft" in self.phases:
                ttft_end = self.phases["ttft"]["end"]
                gen_duration = time.time() - ttft_end
                self.phases["generation"] = {"duration": gen_duration, "start": ttft_end, "end": time.time()}

        if "generation" in self.phases:
            gen_duration = self.phases["generation"]["duration"]
            word_count = len(generated_text.split())

            if gen_duration > 0:
                speed = word_count / gen_duration
                self.metadata["word_count"] = word_count
                self.metadata["speed"] = speed

    def get_total_time(self) -> float:
        """
        Get total elapsed time since query start.

        Returns:
            Total time in seconds
        """
        if self.query_start_time is None:
            return 0.0
        return time.time() - self.query_start_time

    def to_dict(self) -> dict:
        """
        Export all metrics as a dictionary.

        Returns:
            Dictionary containing:
            - Phase durations (by phase name)
            - Derived metrics (speed, word_count, etc.)
            - Total elapsed time

        Example:
            {
                "retrieval": 0.52,
                "ttft": 7.89,
                "generation": 7.32,
                "speed": 14.9,
                "word_count": 109,
                "total_time": 15.73,
                "reranker_status": True  # from metadata
            }
        """
        result = {}

        # Add phase durations
        for phase_name, phase_data in self.phases.items():
            result[phase_name] = phase_data["duration"]

            # Include phase metadata if present
            if "metadata" in phase_data:
                for key, value in phase_data["metadata"].items():
                    result[key] = value

        # Add derived metrics
        result.update(self.metadata)

        # Add total time
        result["total_time"] = self.get_total_time()

        return result
