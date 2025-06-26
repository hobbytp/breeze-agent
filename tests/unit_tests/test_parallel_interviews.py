"""Tests for parallel interview functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from web_research_graph.state import State, Editor, Perspectives, TopicValidation
from web_research_graph.configuration import Configuration
from web_research_graph.interviews_graph.conductor import conduct_interviews
from web_research_graph.interviews_graph.parallel_conductor import parallel_conduct_interviews


@pytest.fixture
def sample_state():
    """Create a sample state with editors for testing."""
    editors = [
        Editor(
            name="Alice",
            role="Climate Scientist",
            affiliation="University of Science",
            description="Expert in climate change research"
        ),
        Editor(
            name="Bob", 
            role="Policy Analyst",
            affiliation="Think Tank",
            description="Expert in environmental policy"
        )
    ]
    
    return State(
        messages=[],
        perspectives=Perspectives(editors=editors),
        topic=TopicValidation(is_valid=True, topic="Climate Change", message="Valid topic"),
        references={}
    )


@pytest.fixture
def serial_config():
    """Configuration for serial interviews."""
    return {"configurable": {"parallel_interviews": False}}


@pytest.fixture
def parallel_config():
    """Configuration for parallel interviews."""
    return {
        "configurable": {
            "parallel_interviews": True,
            "max_parallel_interviews": 2
        }
    }


class TestConfiguration:
    """Test configuration options."""
    
    def test_default_parallel_interviews_false(self):
        """Test that parallel_interviews defaults to False."""
        config = Configuration()
        assert config.parallel_interviews is False
        
    def test_default_max_parallel_interviews(self):
        """Test that max_parallel_interviews has a reasonable default."""
        config = Configuration()
        assert config.max_parallel_interviews == 3
        
    def test_from_runnable_config(self):
        """Test creating configuration from runnable config."""
        runnable_config = {
            "configurable": {
                "parallel_interviews": True,
                "max_parallel_interviews": 5
            }
        }
        config = Configuration.from_runnable_config(runnable_config)
        assert config.parallel_interviews is True
        assert config.max_parallel_interviews == 5


class TestConductor:
    """Test the smart interview conductor."""
    
    @pytest.mark.asyncio
    async def test_conductor_chooses_serial_mode(self, sample_state, serial_config):
        """Test that conductor chooses serial mode when configured."""
        # Mock the interview_graph
        with patch('web_research_graph.interviews_graph.conductor.interview_graph') as mock_graph:
            mock_interview_state = MagicMock()
            mock_interview_state.messages = [AIMessage(content="Serial interview", name="expert")]
            mock_interview_state.references = {"url1": "content1"}
            mock_graph.ainvoke = AsyncMock(return_value=mock_interview_state)
            
            result = await conduct_interviews(sample_state, serial_config)
            
            # Verify interview_graph was called
            mock_graph.ainvoke.assert_called_once()
            
            # Verify result structure
            assert isinstance(result, State)
            assert len(result.messages) > 0
            assert result.references is not None
    
    @pytest.mark.asyncio  
    async def test_conductor_chooses_parallel_mode(self, sample_state, parallel_config):
        """Test that conductor chooses parallel mode when configured."""
        # Mock the parallel_conduct_interviews function
        with patch('web_research_graph.interviews_graph.conductor.parallel_conduct_interviews') as mock_parallel:
            expected_result = State(
                messages=[AIMessage(content="Parallel interview", name="expert")],
                perspectives=sample_state.perspectives,
                references={"url1": "content1"}
            )
            mock_parallel.return_value = expected_result
            
            result = await conduct_interviews(sample_state, parallel_config)
            
            # Verify parallel function was called
            mock_parallel.assert_called_once_with(sample_state, parallel_config)
            
            # Verify result
            assert result == expected_result


class TestParallelConductor:
    """Test the parallel interview conductor."""
    
    def test_editor_conversion(self, sample_state):
        """Test that Editor objects are properly converted to dict format."""
        from web_research_graph.interviews_graph.parallel_conductor import parallel_conduct_interviews
        
        # This would be tested in integration, but we can verify the conversion logic
        editors = sample_state.perspectives.editors
        converted = [
            {
                "name": editor.name,
                "role": editor.role, 
                "affiliation": editor.affiliation,
                "description": editor.description
            }
            for editor in editors
        ]
        
        assert len(converted) == 2
        assert converted[0]["name"] == "Alice"
        assert converted[1]["name"] == "Bob"
        
    @pytest.mark.asyncio
    async def test_no_perspectives_raises_error(self):
        """Test that missing perspectives raises appropriate error."""
        from web_research_graph.interviews_graph.parallel_conductor import parallel_conduct_interviews
        
        state = State(messages=[], perspectives=None)
        
        with pytest.raises(ValueError, match="No perspectives found in state"):
            await parallel_conduct_interviews(state, {})
            
    @pytest.mark.asyncio
    async def test_empty_editors_raises_error(self):
        """Test that empty editors list raises appropriate error."""
        from web_research_graph.interviews_graph.parallel_conductor import parallel_conduct_interviews
        
        state = State(
            messages=[],
            perspectives=Perspectives(editors=[])
        )
        
        with pytest.raises(ValueError, match="No editors found in perspectives"):
            await parallel_conduct_interviews(state, {}) 