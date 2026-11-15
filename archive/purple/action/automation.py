import pyautogui
import os
import subprocess

class DesktopAutomation:
    """
    Controls mouse and keyboard, with a sandbox for self-healing code.
    """
    def __init__(self):
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort

    def type_text(self, text):
        pyautogui.write(text, interval=0.05)

    def click_at(self, x, y):
        pyautogui.click(x, y)

    def execute_script(self, code):
        """
        Executes a generated python script.
        Returns (success, output_or_error)
        """
        temp_file = "temp_action.py"
        with open(temp_file, "w") as f:
            f.write("import pyautogui\nimport time\n")
            f.write(code)
        
        try:
            result = subprocess.run(["python3", temp_file], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
