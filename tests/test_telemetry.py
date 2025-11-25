"""
Tests for telemetry utilities.
"""

import pytest
import time
from src.telemetry import QueryTelemetry


class TestQueryTelemetry:
    """Tests for query telemetry tracking."""

    def test_basic_phase_timing(self):
        """Test basic phase start/end timing."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("test_phase")
        time.sleep(0.01)  # Small delay
        telemetry.end_phase("test_phase")

        metrics = telemetry.to_dict()
        assert "test_phase" in metrics
        assert metrics["test_phase"] >= 0.01
        assert metrics["test_phase"] < 0.1  # Reasonable upper bound

    def test_auto_start_query(self):
        """If start_query() is forgotten, first phase should auto-start it."""
        telemetry = QueryTelemetry()
        # Don't call start_query()

        telemetry.start_phase("retrieval")
        telemetry.end_phase("retrieval")

        assert telemetry.query_start_time is not None
        assert "retrieval" in telemetry.to_dict()

    def test_multiple_phases(self):
        """Test tracking multiple sequential phases."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("phase1")
        time.sleep(0.01)
        telemetry.end_phase("phase1")

        telemetry.start_phase("phase2")
        time.sleep(0.01)
        telemetry.end_phase("phase2")

        metrics = telemetry.to_dict()
        assert "phase1" in metrics
        assert "phase2" in metrics
        assert metrics["phase1"] > 0
        assert metrics["phase2"] > 0

    def test_phase_metadata(self):
        """Test attaching metadata to phases."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("retrieval")
        telemetry.end_phase("retrieval", metadata={"reranker": True, "pool_size": 25})

        metrics = telemetry.to_dict()
        assert metrics["reranker"] is True
        assert metrics["pool_size"] == 25

    def test_implicit_phase_end(self):
        """Test ending phase without explicitly naming it."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("current")
        time.sleep(0.01)
        telemetry.end_phase()  # No name provided - should use current

        metrics = telemetry.to_dict()
        assert "current" in metrics
        assert metrics["current"] > 0

    def test_mark_ttft(self):
        """Test the TTFT convenience method."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("llm_connection")
        time.sleep(0.01)
        telemetry.mark_ttft()

        metrics = telemetry.to_dict()
        assert "ttft" in metrics
        assert metrics["ttft"] > 0

    def test_generation_metrics(self):
        """Test calculation of generation speed metrics."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("generation")
        time.sleep(0.1)
        telemetry.end_phase("generation")

        # 10 words should give speed of ~100 words/sec at 0.1s
        text = "one two three four five six seven eight nine ten"
        telemetry.calculate_generation_metrics(text)

        metrics = telemetry.to_dict()
        assert "speed" in metrics
        assert "word_count" in metrics
        assert metrics["word_count"] == 10
        assert 50 < metrics["speed"] < 150  # Reasonable range given timing variance

    def test_generation_metrics_from_ttft(self):
        """Test generation metrics calculated from TTFT if no explicit generation phase."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        telemetry.start_phase("connection")
        time.sleep(0.01)
        telemetry.mark_ttft()

        time.sleep(0.1)  # Simulate generation time

        text = "generated text here"
        telemetry.calculate_generation_metrics(text)

        metrics = telemetry.to_dict()
        assert "generation" in metrics
        assert "speed" in metrics
        assert metrics["generation"] >= 0.1

    def test_total_time(self):
        """Test total elapsed time calculation."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        time.sleep(0.02)

        telemetry.start_phase("phase1")
        time.sleep(0.01)
        telemetry.end_phase("phase1")

        total = telemetry.get_total_time()
        assert total >= 0.03  # At least sum of sleeps

        metrics = telemetry.to_dict()
        assert "total_time" in metrics
        assert metrics["total_time"] >= 0.03

    def test_flexible_phase_names(self):
        """Test that any string can be used as a phase name."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        custom_names = ["my-custom-phase", "with_underscores", "123numeric", "🚀 emoji"]

        for name in custom_names:
            telemetry.start_phase(name)
            telemetry.end_phase(name)

        metrics = telemetry.to_dict()
        for name in custom_names:
            assert name in metrics

    def test_error_no_active_phase(self):
        """Test error when ending phase with no active phase."""
        telemetry = QueryTelemetry()

        with pytest.raises(ValueError, match="No active phase"):
            telemetry.end_phase()

    def test_error_phase_never_started(self):
        """Test error when ending a phase that was never started."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        with pytest.raises(ValueError, match="never started"):
            telemetry.end_phase("nonexistent_phase")

    def test_to_dict_empty(self):
        """Test to_dict() on empty telemetry."""
        telemetry = QueryTelemetry()
        metrics = telemetry.to_dict()

        assert isinstance(metrics, dict)
        assert metrics["total_time"] == 0.0

    def test_realistic_rag_pipeline(self):
        """Test a realistic RAG query flow."""
        telemetry = QueryTelemetry()
        telemetry.start_query()

        # Phase 1: Retrieval
        telemetry.start_phase("retrieval")
        time.sleep(0.02)
        telemetry.end_phase("retrieval", metadata={"reranker": True})

        # Phase 2: LLM Connection
        telemetry.start_phase("llm_connection")
        time.sleep(0.01)
        telemetry.mark_ttft()

        # Phase 3: Generation (automatic via calculate_generation_metrics)
        time.sleep(0.05)
        response = "This is a sample response with several words in it."
        telemetry.calculate_generation_metrics(response)

        metrics = telemetry.to_dict()

        # Verify all expected metrics
        assert "retrieval" in metrics
        assert "ttft" in metrics
        assert "generation" in metrics
        assert "speed" in metrics
        assert "word_count" in metrics
        assert "reranker" in metrics
        assert "total_time" in metrics

        # Verify relationships
        assert metrics["total_time"] >= metrics["retrieval"] + metrics["ttft"] + metrics["generation"]
        assert metrics["word_count"] == len(response.split())
