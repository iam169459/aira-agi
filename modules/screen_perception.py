"""
Screen Perception Module
Captures and analyzes screen content
"""
import numpy as np
from typing import Optional, Dict, Any

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class ScreenCapture:
    """Captures screen content"""
    
    def __init__(self, monitor=None):
        """
        Initialize screen capture
        monitor: None for full screen, or dict with {top, left, width, height}
        """
        self.monitor = monitor
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture screen image
        Returns: numpy array (H, W, 3) in RGB format, or None on error
        """
        try:
            if HAS_MSS:
                return self._capture_with_mss()
            else:
                # Fallback: return None and let caller handle it
                print("No screen capture library available (install mss)")
                return None
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def _capture_with_mss(self) -> Optional[np.ndarray]:
        """Capture screen using mss library"""
        try:
            with mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]  # Index 1 is the primary monitor
                
                if self.monitor:
                    # Capture specific region
                    x = self.monitor.get('left', monitor['left'])
                    y = self.monitor.get('top', monitor['top'])
                    w = self.monitor.get('width', monitor['width'])
                    h = self.monitor.get('height', monitor['height'])
                    region = {'top': y, 'left': x, 'width': w, 'height': h}
                    sct_img = sct.grab(region)
                else:
                    # Capture full screen
                    sct_img = sct.grab(monitor)
                
                # Convert to numpy array (BGR to RGB)
                img = np.array(sct_img)
                img = img[:, :, :3]  # Remove alpha channel if present
                
                # mss returns BGRA, convert to RGB
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                
                # Convert BGR to RGB (mss captures in BGR format)
                img = img[:, :, ::-1]
                
                return img
        except Exception as e:
            print(f"mSS capture error: {e}")
            return None


class ContentAnalyzer:
    """Analyzes screen content for context"""
    
    def __init__(self):
        pass
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze screen content
        Returns dict with: text_elements, objects, layout_info, etc.
        """
        analysis = {
            'text_elements': [],
            'objects': [],
            'dominant_colors': [],
            'overall_context': 'unknown'
        }
        
        if image is None:
            return analysis
        
        # Placeholder for actual analysis
        # Would use OCR (tesseract), object detection, etc.
        
        return analysis
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        # Placeholder - would use pytesseract
        return ""


class PerceptionState:
    """Persistent perception state across sessions"""
    
    def __init__(self):
        self.last_image = None
        self.last_analysis = {}
        self.text_history = []
        self.object_history = []
    
    def update(self, image: np.ndarray, analysis: Dict):
        """Update perception state"""
        self.last_image = image
        self.last_analysis = analysis
        self.text_history.append(analysis.get('text_elements', [])[-1:] if analysis else [])
        self.object_history.append(analysis.get('objects', [])[-1:] if analysis else [])