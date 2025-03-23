"""
Keyboard shortcut management system for the application.

This module provides a centralized system for managing keyboard shortcuts,
including default shortcuts, customization, and conflict resolution.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
from dataclasses import dataclass
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QKeySequence, QAction, QShortcut

logger = logging.getLogger(__name__)


class ShortcutCategory(Enum):
    """Categories for organizing shortcuts."""
    FILE = auto()
    EDIT = auto()
    VIEW = auto()
    IMAGE = auto()
    LAYER = auto()
    SELECTION = auto()
    TOOLS = auto()
    FILTER = auto()
    HELP = auto()


@dataclass
class ShortcutInfo:
    """Information about a keyboard shortcut."""
    id: str
    category: ShortcutCategory
    title: str
    description: str
    default_sequence: str
    custom_sequence: Optional[str] = None
    action: Optional[QAction] = None
    callback: Optional[Callable] = None
    
    @property
    def sequence(self) -> str:
        """Get the effective key sequence (custom or default)."""
        return self.custom_sequence or self.default_sequence


class ShortcutManager(QObject):
    """Manager for keyboard shortcuts throughout the application."""
    
    shortcuts_changed = pyqtSignal()  # Emitted when shortcuts are changed
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the shortcut manager.
        
        Args:
            config_path: Path to the shortcuts configuration file
        """
        super().__init__()
        
        # Shortcuts dictionary, keyed by ID
        self._shortcuts: Dict[str, ShortcutInfo] = {}
        
        # Actions that are assigned shortcuts
        self._actions: Dict[str, QAction] = {}
        
        # Path to configuration file
        self._config_path = config_path or os.path.expanduser("~/.helixzone/shortcuts.json")
        
        # Initialize default shortcuts
        self._init_default_shortcuts()
        
        # Load custom shortcuts
        self._load_custom_shortcuts()
    
    def _init_default_shortcuts(self):
        """Initialize the default shortcuts."""
        # File operations
        self.register_shortcut(
            "file_new", 
            ShortcutCategory.FILE, 
            "New", 
            "Create a new image", 
            "Ctrl+N"
        )
        self.register_shortcut(
            "file_open", 
            ShortcutCategory.FILE, 
            "Open", 
            "Open an existing image", 
            "Ctrl+O"
        )
        self.register_shortcut(
            "file_save", 
            ShortcutCategory.FILE, 
            "Save", 
            "Save the current image", 
            "Ctrl+S"
        )
        self.register_shortcut(
            "file_save_as", 
            ShortcutCategory.FILE, 
            "Save As", 
            "Save the current image with a new name", 
            "Ctrl+Shift+S"
        )
        self.register_shortcut(
            "file_export", 
            ShortcutCategory.FILE, 
            "Export", 
            "Export the image to a different format", 
            "Ctrl+E"
        )
        self.register_shortcut(
            "file_close", 
            ShortcutCategory.FILE, 
            "Close", 
            "Close the current image", 
            "Ctrl+W"
        )
        self.register_shortcut(
            "file_quit", 
            ShortcutCategory.FILE, 
            "Quit", 
            "Quit the application", 
            "Ctrl+Q"
        )
        
        # Edit operations
        self.register_shortcut(
            "edit_undo", 
            ShortcutCategory.EDIT, 
            "Undo", 
            "Undo the last operation", 
            "Ctrl+Z"
        )
        self.register_shortcut(
            "edit_redo", 
            ShortcutCategory.EDIT, 
            "Redo", 
            "Redo the previously undone operation", 
            "Ctrl+Shift+Z"
        )
        self.register_shortcut(
            "edit_cut", 
            ShortcutCategory.EDIT, 
            "Cut", 
            "Cut the selected content", 
            "Ctrl+X"
        )
        self.register_shortcut(
            "edit_copy", 
            ShortcutCategory.EDIT, 
            "Copy", 
            "Copy the selected content", 
            "Ctrl+C"
        )
        self.register_shortcut(
            "edit_paste", 
            ShortcutCategory.EDIT, 
            "Paste", 
            "Paste content from clipboard", 
            "Ctrl+V"
        )
        self.register_shortcut(
            "edit_paste_as_new_layer", 
            ShortcutCategory.EDIT, 
            "Paste as New Layer", 
            "Paste content from clipboard as a new layer", 
            "Ctrl+Shift+V"
        )
        self.register_shortcut(
            "edit_select_all", 
            ShortcutCategory.EDIT, 
            "Select All", 
            "Select the entire canvas", 
            "Ctrl+A"
        )
        self.register_shortcut(
            "edit_deselect", 
            ShortcutCategory.EDIT, 
            "Deselect", 
            "Clear the current selection", 
            "Ctrl+D"
        )
        
        # View operations
        self.register_shortcut(
            "view_zoom_in", 
            ShortcutCategory.VIEW, 
            "Zoom In", 
            "Zoom in on the image", 
            "Ctrl++"
        )
        self.register_shortcut(
            "view_zoom_out", 
            ShortcutCategory.VIEW, 
            "Zoom Out", 
            "Zoom out from the image", 
            "Ctrl+-"
        )
        self.register_shortcut(
            "view_zoom_fit", 
            ShortcutCategory.VIEW, 
            "Zoom to Fit", 
            "Zoom to fit the entire image in view", 
            "Ctrl+0"
        )
        self.register_shortcut(
            "view_zoom_actual", 
            ShortcutCategory.VIEW, 
            "Actual Size", 
            "View the image at its actual size (100%)", 
            "Ctrl+1"
        )
        self.register_shortcut(
            "view_toggle_grid", 
            ShortcutCategory.VIEW, 
            "Toggle Grid", 
            "Show or hide the grid", 
            "Ctrl+'"
        )
        self.register_shortcut(
            "view_toggle_rulers", 
            ShortcutCategory.VIEW, 
            "Toggle Rulers", 
            "Show or hide the rulers", 
            "Ctrl+R"
        )
        
        # Layer operations
        self.register_shortcut(
            "layer_new", 
            ShortcutCategory.LAYER, 
            "New Layer", 
            "Create a new layer", 
            "Ctrl+Shift+N"
        )
        self.register_shortcut(
            "layer_duplicate", 
            ShortcutCategory.LAYER, 
            "Duplicate Layer", 
            "Duplicate the current layer", 
            "Ctrl+J"
        )
        self.register_shortcut(
            "layer_delete", 
            ShortcutCategory.LAYER, 
            "Delete Layer", 
            "Delete the current layer", 
            "Shift+Delete"
        )
        self.register_shortcut(
            "layer_merge_down", 
            ShortcutCategory.LAYER, 
            "Merge Down", 
            "Merge the current layer with the one below", 
            "Ctrl+E"
        )
        self.register_shortcut(
            "layer_flatten", 
            ShortcutCategory.LAYER, 
            "Flatten Image", 
            "Flatten all layers into one", 
            "Ctrl+Shift+E"
        )
        
        # Selection operations
        self.register_shortcut(
            "selection_invert", 
            ShortcutCategory.SELECTION, 
            "Invert Selection", 
            "Invert the current selection", 
            "Ctrl+Shift+I"
        )
        self.register_shortcut(
            "selection_feather", 
            ShortcutCategory.SELECTION, 
            "Feather Selection", 
            "Feather the edges of the current selection", 
            "Ctrl+Alt+D"
        )
        self.register_shortcut(
            "selection_grow", 
            ShortcutCategory.SELECTION, 
            "Grow Selection", 
            "Expand the current selection", 
            "Ctrl+Alt+."
        )
        self.register_shortcut(
            "selection_shrink", 
            ShortcutCategory.SELECTION, 
            "Shrink Selection", 
            "Contract the current selection", 
            "Ctrl+Alt+,"
        )
        
        # Tool operations
        self.register_shortcut(
            "tool_move", 
            ShortcutCategory.TOOLS, 
            "Move Tool", 
            "Select the move tool", 
            "V"
        )
        self.register_shortcut(
            "tool_rectangular_selection", 
            ShortcutCategory.TOOLS, 
            "Rectangular Selection", 
            "Select the rectangular selection tool", 
            "M"
        )
        self.register_shortcut(
            "tool_elliptical_selection", 
            ShortcutCategory.TOOLS, 
            "Elliptical Selection", 
            "Select the elliptical selection tool", 
            "E"
        )
        self.register_shortcut(
            "tool_lasso", 
            ShortcutCategory.TOOLS, 
            "Lasso Tool", 
            "Select the lasso selection tool", 
            "L"
        )
        self.register_shortcut(
            "tool_brush", 
            ShortcutCategory.TOOLS, 
            "Brush Tool", 
            "Select the brush tool", 
            "B"
        )
        self.register_shortcut(
            "tool_eraser", 
            ShortcutCategory.TOOLS, 
            "Eraser Tool", 
            "Select the eraser tool", 
            "Shift+E"
        )
        self.register_shortcut(
            "tool_fill", 
            ShortcutCategory.TOOLS, 
            "Fill Tool", 
            "Select the fill tool", 
            "G"
        )
        self.register_shortcut(
            "tool_text", 
            ShortcutCategory.TOOLS, 
            "Text Tool", 
            "Select the text tool", 
            "T"
        )
        self.register_shortcut(
            "tool_crop", 
            ShortcutCategory.TOOLS, 
            "Crop Tool", 
            "Select the crop tool", 
            "C"
        )
        self.register_shortcut(
            "tool_hand", 
            ShortcutCategory.TOOLS, 
            "Hand Tool", 
            "Select the hand (pan) tool", 
            "H"
        )
        self.register_shortcut(
            "tool_zoom", 
            ShortcutCategory.TOOLS, 
            "Zoom Tool", 
            "Select the zoom tool", 
            "Z"
        )
        
        # Filter operations
        self.register_shortcut(
            "filter_repeat_last", 
            ShortcutCategory.FILTER, 
            "Repeat Last Filter", 
            "Repeat the last applied filter", 
            "Ctrl+F"
        )
        
        # Help operations
        self.register_shortcut(
            "help_show", 
            ShortcutCategory.HELP, 
            "Show Help", 
            "Show the help documentation", 
            "F1"
        )
    
    def register_shortcut(self, 
                        shortcut_id: str, 
                        category: ShortcutCategory, 
                        title: str, 
                        description: str, 
                        default_sequence: str,
                        callback: Optional[Callable] = None) -> ShortcutInfo:
        """Register a new shortcut.
        
        Args:
            shortcut_id: Unique identifier for the shortcut
            category: Category of the shortcut
            title: User-friendly name for the shortcut
            description: Description of what the shortcut does
            default_sequence: Default key sequence as string (e.g. "Ctrl+C")
            callback: Optional function to call when shortcut is triggered
            
        Returns:
            The registered shortcut info
        """
        shortcut = ShortcutInfo(
            id=shortcut_id,
            category=category,
            title=title,
            description=description,
            default_sequence=default_sequence,
            callback=callback
        )
        
        self._shortcuts[shortcut_id] = shortcut
        return shortcut
    
    def get_shortcut(self, shortcut_id: str) -> Optional[ShortcutInfo]:
        """Get a shortcut by ID.
        
        Args:
            shortcut_id: The ID of the shortcut to get
            
        Returns:
            The shortcut info, or None if not found
        """
        return self._shortcuts.get(shortcut_id)
    
    def get_all_shortcuts(self) -> List[ShortcutInfo]:
        """Get all registered shortcuts.
        
        Returns:
            List of all shortcut info objects
        """
        return list(self._shortcuts.values())
    
    def get_shortcuts_by_category(self, category: ShortcutCategory) -> List[ShortcutInfo]:
        """Get shortcuts filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of shortcuts in the category
        """
        return [s for s in self._shortcuts.values() if s.category == category]
    
    def set_custom_shortcut(self, shortcut_id: str, sequence: str) -> bool:
        """Set a custom key sequence for a shortcut.
        
        Args:
            shortcut_id: ID of the shortcut to customize
            sequence: New key sequence as string
            
        Returns:
            True if successful, False if the shortcut doesn't exist
        """
        shortcut = self.get_shortcut(shortcut_id)
        if not shortcut:
            return False
            
        # Check for conflicts
        existing = self._find_shortcut_by_sequence(sequence)
        if existing and existing.id != shortcut_id:
            logger.warning(f"Shortcut conflict: {sequence} already used by {existing.title}")
            return False
            
        shortcut.custom_sequence = sequence
        
        # Update the action if it exists
        if shortcut.action:
            shortcut.action.setShortcut(QKeySequence(sequence))
        
        # Save custom shortcuts
        self._save_custom_shortcuts()
        
        # Emit change signal
        self.shortcuts_changed.emit()
        
        return True
    
    def reset_shortcut(self, shortcut_id: str) -> bool:
        """Reset a shortcut to its default key sequence.
        
        Args:
            shortcut_id: ID of the shortcut to reset
            
        Returns:
            True if successful, False if the shortcut doesn't exist
        """
        shortcut = self.get_shortcut(shortcut_id)
        if not shortcut:
            return False
            
        shortcut.custom_sequence = None
        
        # Update the action if it exists
        if shortcut.action:
            shortcut.action.setShortcut(QKeySequence(shortcut.default_sequence))
        
        # Save custom shortcuts
        self._save_custom_shortcuts()
        
        # Emit change signal
        self.shortcuts_changed.emit()
        
        return True
    
    def reset_all_shortcuts(self) -> None:
        """Reset all shortcuts to their default key sequences."""
        for shortcut in self._shortcuts.values():
            shortcut.custom_sequence = None
            
            # Update the action if it exists
            if shortcut.action:
                shortcut.action.setShortcut(QKeySequence(shortcut.default_sequence))
        
        # Save custom shortcuts
        self._save_custom_shortcuts()
        
        # Emit change signal
        self.shortcuts_changed.emit()
    
    def connect_action(self, shortcut_id: str, action: QAction) -> bool:
        """Connect a shortcut to a QAction.
        
        Args:
            shortcut_id: ID of the shortcut to connect
            action: QAction to connect to
            
        Returns:
            True if successful, False if the shortcut doesn't exist
        """
        shortcut = self.get_shortcut(shortcut_id)
        if not shortcut:
            return False
            
        # Set the action's shortcut
        action.setShortcut(QKeySequence(shortcut.sequence))
        
        # Store the action reference
        shortcut.action = action
        self._actions[shortcut_id] = action
        
        return True
    
    def connect_callback(self, shortcut_id: str, callback: Callable) -> bool:
        """Connect a shortcut to a callback function.
        
        Args:
            shortcut_id: ID of the shortcut to connect
            callback: Function to call when shortcut is triggered
            
        Returns:
            True if successful, False if the shortcut doesn't exist
        """
        shortcut = self.get_shortcut(shortcut_id)
        if not shortcut:
            return False
            
        # Store the callback
        shortcut.callback = callback
        
        return True
    
    def _load_custom_shortcuts(self) -> None:
        """Load custom shortcuts from configuration file."""
        if not os.path.exists(self._config_path):
            return
            
        try:
            with open(self._config_path, 'r') as f:
                data = json.load(f)
                
            # Apply custom shortcuts
            for shortcut_id, sequence in data.items():
                shortcut = self.get_shortcut(shortcut_id)
                if shortcut:
                    shortcut.custom_sequence = sequence
                    
        except Exception as e:
            logger.error(f"Error loading custom shortcuts: {e}")
    
    def _save_custom_shortcuts(self) -> None:
        """Save custom shortcuts to configuration file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        
        # Build dictionary of custom shortcuts
        custom_shortcuts = {}
        for shortcut in self._shortcuts.values():
            if shortcut.custom_sequence:
                custom_shortcuts[shortcut.id] = shortcut.custom_sequence
                
        try:
            with open(self._config_path, 'w') as f:
                json.dump(custom_shortcuts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving custom shortcuts: {e}")
    
    def _find_shortcut_by_sequence(self, sequence: str) -> Optional[ShortcutInfo]:
        """Find a shortcut by its key sequence.
        
        Args:
            sequence: Key sequence to search for
            
        Returns:
            The matching shortcut, or None if not found
        """
        normalized = QKeySequence(sequence).toString()
        for shortcut in self._shortcuts.values():
            if QKeySequence(shortcut.sequence).toString() == normalized:
                return shortcut
        return None


# Singleton pattern
_shortcut_manager = None

def get_shortcut_manager() -> ShortcutManager:
    """Get the global shortcut manager instance.
    
    Returns:
        The shortcut manager instance
    """
    global _shortcut_manager
    if _shortcut_manager is None:
        _shortcut_manager = ShortcutManager()
    return _shortcut_manager 