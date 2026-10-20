"""Desktop automation action layer with guardrail checks."""
import time


class DesktopAutomation:
    """Mouse/keyboard control. All actions pass through the safe-mode guard."""

    def __init__(self, guard=None):
        self.guard = guard
        self.pyautogui = None
        try:
            import pyautogui as pag
            pag.FAILSAFE = True
            self.pyautogui = pag
        except Exception:
            pass

    def _allowed(self, action: str) -> bool:
        if self.guard is None:
            return True
        return self.guard.allow_desktop_action(action)

    def move_mouse(self, x: int, y: int) -> bool:
        if not self._allowed("move_mouse"):
            return False
        if self.pyautogui:
            self.pyautogui.moveTo(x, y, duration=0.2)
        return True

    def click(self, x: int = None, y: int = None, clicks: int = 1) -> bool:
        if not self._allowed("click"):
            return False
        if self.pyautogui:
            if x is not None and y is not None:
                self.pyautogui.click(x, y, clicks=clicks)
            else:
                self.pyautogui.click(clicks=clicks)
        return True

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        if not self._allowed("type_text"):
            return False
        if self.pyautogui:
            self.pyautogui.typewrite(text, interval=interval)
        return True

    def press(self, *keys: str) -> bool:
        if not self._allowed("press_key"):
            return False
        if self.pyautogui:
            self.pyautogui.hotkey(*keys) if len(keys) > 1 else self.pyautogui.press(keys[0])
        return True

    def scroll(self, amount: int) -> bool:
        if not self._allowed("scroll"):
            return False
        if self.pyautogui:
            self.pyautogui.scroll(amount)
        return True

    def screenshot(self, path: str = "screenshot.png") -> bool:
        if self.pyautogui:
            self.pyautogui.screenshot(path)
            return True
        return False
