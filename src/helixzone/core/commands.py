from PyQt6.QtGui import QImage
from PyQt6.QtCore import QObject, pyqtSignal

class Command:
    """Base class for all undoable commands."""
    def __init__(self, description=""):
        self.description = description
    
    def execute(self):
        """Execute the command."""
        pass
    
    def undo(self):
        """Undo the command."""
        pass
    
    def redo(self):
        """Redo the command. By default, just executes again."""
        self.execute()

class DrawCommand(Command):
    """Command for drawing operations."""
    def __init__(self, layer, image_before):
        super().__init__("Draw")
        self.layer = layer
        self.image_before = image_before.copy() if image_before is not None else None
        self.image_after = None
    
    def execute(self):
        """Execute is called after the drawing is complete."""
        if self.layer and self.layer.image:
            self.image_after = self.layer.image.copy()
    
    def undo(self):
        """Restore the image state from before the drawing."""
        if self.layer and self.image_before:
            self.layer.image = self.image_before.copy()
            self.layer.changed.emit()
    
    def redo(self):
        """Restore the image state from after the drawing."""
        if self.layer and self.image_after:
            self.layer.image = self.image_after.copy()
            self.layer.changed.emit()

class LayerCommand(Command):
    """Command for layer operations (add, remove, move)."""
    def __init__(self, layer_stack, description, undo_func, redo_func):
        super().__init__(description)
        self.layer_stack = layer_stack
        self.undo_func = undo_func
        self.redo_func = redo_func
    
    def execute(self):
        self.redo_func()
    
    def undo(self):
        self.undo_func()
    
    def redo(self):
        self.redo_func()

class CommandStack(QObject):
    """Manages the undo/redo stack."""
    
    changed = pyqtSignal()  # Emitted when the stack changes
    
    def __init__(self):
        super().__init__()
        self.undo_stack = []
        self.redo_stack = []
        self.is_executing = False  # Prevent recursive execution
    
    def push(self, command):
        """Push a new command onto the stack."""
        if self.is_executing:
            return
            
        self.is_executing = True
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()  # Clear redo stack when new command is added
        self.is_executing = False
        self.changed.emit()
    
    def undo(self):
        """Undo the last command."""
        if not self.undo_stack or self.is_executing:
            return
            
        self.is_executing = True
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.is_executing = False
        self.changed.emit()
    
    def redo(self):
        """Redo the last undone command."""
        if not self.redo_stack or self.is_executing:
            return
            
        self.is_executing = True
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        self.is_executing = False
        self.changed.emit()
    
    def can_undo(self):
        """Check if there are commands to undo."""
        return len(self.undo_stack) > 0
    
    def can_redo(self):
        """Check if there are commands to redo."""
        return len(self.redo_stack) > 0
    
    def get_undo_text(self):
        """Get the description of the next undo command."""
        if self.can_undo():
            return f"Undo {self.undo_stack[-1].description}"
        return "Undo"
    
    def get_redo_text(self):
        """Get the description of the next redo command."""
        if self.can_redo():
            return f"Redo {self.redo_stack[-1].description}"
        return "Redo" 