"""State management module with comprehensive debugging and validation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, TypeVar, Union
from dataclasses import dataclass, field
import json
import time
from copy import deepcopy
from ..core.debugger import ProjectDebugger

# Type aliases
T = TypeVar('T')
StateDict = Dict[str, Any]
ValidationFunc = Callable[[Any], bool]
RangeType = Tuple[Union[int, float], Union[int, float]]

# Get the global debugger instance
debugger = ProjectDebugger(
    project_name="helixzone",
    log_dir="debug_logs"
)

@dataclass
class StateHistory:
    """Tracks state changes over time."""
    changes: List[StateDict] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

@dataclass
class StateValidation:
    """Validation rules and constraints for state updates."""
    required_fields: Set[str] = field(default_factory=set)
    field_types: Dict[str, type] = field(default_factory=dict)
    value_ranges: Dict[str, RangeType] = field(default_factory=dict)
    custom_validators: Dict[str, ValidationFunc] = field(default_factory=dict)

class StateManager:
    """Manages application state with debugging and validation."""
    
    def __init__(self, initial_state: Optional[StateDict] = None):
        self.debugger = debugger
        self.state: StateDict = initial_state or {}
        self.history = StateHistory()
        self.validation = StateValidation()
        
        # Initialize state
        if initial_state:
            self.debugger.checkpoint("Initializing state manager")
            self._validate_state(initial_state)
            self._log_state_change("initialization", initial_state)
    
    @debugger.trace_function
    def update_state(
        self,
        new_data: StateDict,
        author: str = "system",
        reason: str = "update"
    ) -> None:
        """Update state with validation and logging.
        
        Args:
            new_data: Dictionary of state updates
            author: Name of the component/user making the change
            reason: Reason for the state change
            
        Raises:
            ValueError: If validation fails
            KeyError: If required fields are missing
        """
        with self.debugger.error_boundary("state_update"):
            # Create state snapshot
            old_state = deepcopy(self.state)
            
            # Log state changes
            self.debugger.variable_dump(
                prefix="state_update",
                old_state=old_state,
                new_data=new_data,
                author=author,
                reason=reason
            )
            
            try:
                # Validate new data
                self._validate_state(new_data)
                
                # Update state
                self.state.update(new_data)
                
                # Record change
                self._log_state_change(reason, new_data, author)
                
                # Create checkpoint
                self.debugger.checkpoint(
                    f"State updated: {reason}",
                    include_memory=False
                )
                
            except Exception as e:
                # Restore state on error
                self.state = old_state
                self.debugger.error_logger.error(
                    f"State update failed: {str(e)}, "
                    f"Author: {author}, Reason: {reason}"
                )
                raise
    
    def _validate_state(self, state_data: StateDict) -> None:
        """Validate state updates against defined rules.
        
        Args:
            state_data: State data to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        missing_fields = self.validation.required_fields - set(state_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate types
        for field, expected_type in self.validation.field_types.items():
            if field in state_data and not isinstance(state_data[field], expected_type):
                raise ValueError(
                    f"Invalid type for {field}: "
                    f"expected {expected_type.__name__}, "
                    f"got {type(state_data[field]).__name__}"
                )
        
        # Validate value ranges
        for field, (min_val, max_val) in self.validation.value_ranges.items():
            if field in state_data:
                value = state_data[field]
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"Value out of range for {field}: "
                        f"got {value}, expected [{min_val}, {max_val}]"
                    )
        
        # Run custom validators
        for field, validator in self.validation.custom_validators.items():
            if field in state_data:
                try:
                    validator(state_data[field])
                except Exception as e:
                    raise ValueError(f"Validation failed for {field}: {str(e)}")
    
    def _log_state_change(
        self,
        reason: str,
        changes: StateDict,
        author: str = "system"
    ) -> None:
        """Log a state change to history.
        
        Args:
            reason: Reason for the state change
            changes: Dictionary of changes made
            author: Author of the changes
        """
        self.history.changes.append(deepcopy(changes))
        self.history.timestamps.append(time.time())
        self.history.authors.append(author)
        self.history.reasons.append(reason)
        
        # Log to state logger
        self.debugger.state_logger.info(
            f"State change by {author}: {reason}\n"
            f"Changes: {json.dumps(changes, indent=2)}"
        )
    
    def add_validation_rule(
        self,
        field: str,
        field_type: Optional[type] = None,
        required: bool = False,
        value_range: Optional[RangeType] = None,
        custom_validator: Optional[ValidationFunc] = None
    ) -> None:
        """Add a validation rule for a state field.
        
        Args:
            field: Name of the field to validate
            field_type: Expected type of the field
            required: Whether the field is required
            value_range: Tuple of (min, max) values
            custom_validator: Custom validation function
        """
        if required:
            self.validation.required_fields.add(field)
        
        if field_type is not None:
            self.validation.field_types[field] = field_type
        
        if value_range is not None:
            self.validation.value_ranges[field] = value_range
        
        if custom_validator is not None:
            self.validation.custom_validators[field] = custom_validator
    
    def get_state_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get state change history within a time range.
        
        Args:
            start_time: Start time (timestamp)
            end_time: End time (timestamp)
            
        Returns:
            List of state changes with metadata
        """
        history = []
        for i, timestamp in enumerate(self.history.timestamps):
            if ((start_time is None or timestamp >= start_time) and
                (end_time is None or timestamp <= end_time)):
                history.append({
                    'timestamp': timestamp,
                    'author': self.history.authors[i],
                    'reason': self.history.reasons[i],
                    'changes': self.history.changes[i]
                })
        return history
    
    def get_current_state(self) -> StateDict:
        """Get the current state.
        
        Returns:
            Deep copy of current state
        """
        return deepcopy(self.state)
    
    def reset_state(self, initial_state: Optional[StateDict] = None) -> None:
        """Reset the state manager.
        
        Args:
            initial_state: Optional initial state to set
        """
        self.state = initial_state or {}
        self.history = StateHistory()
        self.debugger.checkpoint("State manager reset") 