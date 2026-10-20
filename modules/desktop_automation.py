"""
Desktop Automation Module
Provides mouse and keyboard control
"""
import keyboard
import time
import numpy as np
from typing import Optional, Tuple, List


class DesktopController:
    """Controls mouse and keyboard for desktop automation"""
    
    def __init__(self, speed=1.0, pause_between_actions=0.1):
        """
        Initialize desktop controller
        speed: multiplier for mouse movement speed
        pause_between_actions: default pause in seconds
        """
        self.FAILSAFE = True
        self.PAUSE = pause_between_actions
        self.speed = speed
        # Default screen size fallback
        self.screen_size = (1920, 1080)
    
    # Mouse operations
    def move_to(self, x: int, y: int, duration: Optional[float] = None):
        """Move mouse to absolute position"""
        duration = duration or (abs(x) + abs(y)) / 1000 * self.speed
        pyautogui.moveTo(x, y, duration=duration)
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = 'left', clicks: int = 1):
        """Click at position or current position"""
        if x is not None and y is not None:
            self.move_to(x, y)
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Double-click at position"""
        if x is not None and y is not None:
            self.move_to(x, y)
        pyautogui.doubleClick(x=x, y=y)
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Right-click at position"""
        if x is not None and y is not None:
            self.move_to(x, y)
        pyautogui.rightClick(x=x, y=y)
    
    def drag_to(self, x: int, y: int, duration: float = 1.0):
        """Drag mouse to position"""
        self.move_to(x, y, duration=duration)
        pyautogui.dragTo(x, y, duration=duration, button='left')
    
    # Keyboard operations
    def type(self, text: str, interval: float = 0.01):
        """Type text"""
        pyautogui.typewrite(text, interval=interval)
    
    def press(self, key: str):
        """Press a single key"""
        pyautogui.press(key)
    
    def write(self, text: str, interval: float = 0.01):
        """Write text (alias for type)"""
        self.type(text, interval)
    
    def press_keyboard(self, key: str, press_duration: float = 0.1):
        """Press and hold a key"""
        keyboard.press(key)
        time.sleep(press_duration)
        keyboard.release(key)
    
    # Convenience methods
    def scroll(self, amount: int):
        """Scroll mouse wheel"""
        pyautogui.scroll(amount)
    
    def press_hotkey(self, *keys: str):
        """Press hotkey combination"""
        pyautogui.hotkey(*keys)


class AutomationSequence:
    """Sequence of automated actions"""
    
    def __init__(self):
        self.steps = []
    
    def add_click(self, x: int, y: int, button: str = 'left', clicks: int = 1):
        """Add click step"""
        self.steps.append(('click', {'x': x, 'y': y, 'button': button, 'clicks': clicks}))
        return self
    
    def add_type(self, text: str, interval: float = 0.01):
        """Add type step"""
        self.steps.append(('type', {'text': text, 'interval': interval}))
        return self
    
    def add_press(self, key: str):
        """Add key press step"""
        self.steps.append(('press', {'key': key}))
        return self
    
    def execute(self, controller: DesktopController):
        """Execute the sequence"""
        for step_type, params in self.steps:
            if step_type == 'click':
                controller.click(**params)
            elif step_type == 'type':
                controller.type(**params)
            elif step_type == 'press':
                controller.press(**params)


class SafeModeGuard:
    """Safe-mode guardrails for desktop automation"""
    
    PROTECTED_AREAS = {
        'cursor_position': None,  # Don't move cursor to certain areas
        'critical_systems': ['task_manager', 'system_settings', 'terminal'],
        'max_clicks_per_minute': 60,
    }
    
    def __init__(self):
        self.clicks_this_minute = 0
        self.last_click_time = time.time()
    
    def can_execute(self, action: str, **kwargs) -> bool:
        """Check if action is safe to execute"""
        # Check protected systems
        if 'click' in action.lower() or 'press' in action.lower():
            current_time = time.time()
            if current_time - self.last_click_time < 60:
                self.clicks_this_minute += 1
            else:
                self.clicks_this_minute = 1
                self.last_click_time = current_time
            
            if self.clicks_this_minute > self.PROTECTED_AREAS['max_clicks_per_minute']:
                print(f"Safe-mode: Too many clicks ({self.clicks_this_minute}/minute)")
                return False
        
        return True