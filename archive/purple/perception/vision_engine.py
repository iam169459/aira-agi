from PIL import ImageGrab
import base64
from io import BytesIO

class VisionEngine:
    """
    Captures and analyzes the desktop screen.
    """
    def capture_screen(self):
        """Takes a screenshot and returns it as a base64 string."""
        screenshot = ImageGrab.grab()
        
        # Shrink image to save tokens/bandwidth
        screenshot.thumbnail((1280, 720))
        
        buffered = BytesIO()
        screenshot.save(buffered, format="JPEG", quality=70)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def describe_screen(self, model_client):
        """Uses a VLM to describe what's happening on screen."""
        image_b64 = self.capture_screen()
        # This will be called by the Brain module with an LLM client
        return image_b64
