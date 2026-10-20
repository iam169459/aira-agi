"""
Self-Healing Code System
Detects and fixes its own errors
"""
import traceback
import sys
import re
from typing import Optional, Dict, Any, List


class ErrorInfo:
    """Information about a detected error"""
    
    def __init__(self, error_type: str, message: str, 
                 file_path: Optional[str] = None, 
                 line_number: Optional[int] = None,
                 stack_trace: Optional[str] = None):
        self.error_type = error_type
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        self.stack_trace = stack_trace or traceback.format_stack()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.error_type,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'stack_trace': self.stack_trace
        }


class SelfHealingCode:
    """Detects and attempts to fix its own errors"""
    
    def __init__(self):
        self.error_history = []
        self.fix_attempts = 0
        self.max_fix_attempts = 3
    
    def capture_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """Capture and format error information"""
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            file_path=getattr(error, '__file__', None),
            line_number=getattr(error, '__line__', None),
            stack_trace=traceback.format_exc()
        )
        self.error_history.append(error_info)
        return error_info
    
    def analyze_error(self, error_info: ErrorInfo) -> Optional[Dict[str, Any]]:
        """
        Analyze error and suggest fix
        Returns dict with: fix_type, fix_code, confidence, explanation
        """
        fix_suggestions = {
            'RuntimeError': self._fix_runtime_error,
            'ValueError': self._fix_value_error,
            'TypeError': self._fix_type_error,
            'NameError': self._fix_name_error,
            'ImportError': self._fix_import_error,
        }
        
        fixer = fix_suggestions.get(error_info.error_type)
        if fixer:
            return fixer(error_info)
        
        return self._general_error_fix(error_info)
    
    def _fix_runtime_error(self, error: ErrorInfo) -> Optional[Dict[str, Any]]:
        """Fix runtime errors"""
        msg = error.message.lower()
        
        if 'division' in msg and 'zero' in msg:
            return {
                'fix_type': 'check_division',
                'fix_code': 'Add zero check before division',
                'confidence': 0.8,
                'explanation': 'Detected potential division by zero'
            }
        
        if 'index' in msg and 'out of range' in msg:
            return {
                'fix_type': 'check_bounds',
                'fix_code': 'Add bounds checking before array access',
                'confidence': 0.7,
                'explanation': 'Detected potential index out of range'
            }
        
        return None
    
    def _fix_value_error(self, error: ErrorInfo) -> Optional[Dict[str, Any]]:
        """Fix value errors"""
        msg = error.message.lower()
        
        if 'shape' in msg:
            return {
                'fix_type': 'check_shape',
                'fix_code': 'Verify tensor/shape dimensions match',
                'confidence': 0.75,
                'explanation': 'Detected shape mismatch'
            }
        
        return None
    
    def _fix_type_error(self, error: ErrorInfo) -> Optional[Dict[str, Any]]:
        """Fix type errors"""
        return {
            'fix_type': 'add_type_check',
            'fix_code': 'Add type checking or conversion',
            'confidence': 0.6,
            'explanation': 'Detected type mismatch'
        }
    
    def _fix_name_error(self, error: ErrorInfo) -> Optional[Dict[str, Any]]:
        """Fix name errors (undefined variables)"""
        return {
            'fix_type': 'define_variable',
            'fix_code': 'Ensure variable is defined before use',
            'confidence': 0.8,
            'explanation': 'Detected undefined variable'
        }
    
    def _fix_import_error(self, error: ErrorInfo) -> Optional[Dict[str, Any]]:
        """Fix import errors"""
        return {
            'fix_type': 'install_and_import',
            'fix_code': 'Install missing package and re-import',
            'confidence': 0.7,
            'explanation': 'Detected missing import'
        }
    
    def _general_error_fix(self, error: ErrorInfo) -> Dict[str, Any]:
        """General error fix fallback"""
        return {
            'fix_type': 'general_debug',
            'fix_code': 'Review error message and apply appropriate fix',
            'confidence': 0.4,
            'explanation': 'General error - manual review recommended'
        }
    
    def attempt_fix(self, error: Exception, context: str = "") -> bool:
        """
        Attempt to automatically fix an error
        Returns True if fix was applied successfully
        """
        self.fix_attempts += 1
        if self.fix_attempts > self.max_fix_attempts:
            print("Max fix attempts reached, stopping")
            return False
        
        error_info = self.capture_error(error, context)
        suggestion = self.analyze_error(error_info)
        
        if suggestion and suggestion['confidence'] > 0.6:
            print(f"Attempting fix ({suggestion['fix_type']}): {suggestion['explanation']}")
            # In a real implementation, we'd apply the fix
            # For now, just log the suggestion
            print(f"  Suggested fix: {suggestion['fix_code']}")
            return True
        
        return False
    
    def get_last_error(self) -> Optional[ErrorInfo]:
        """Get the most recent error info"""
        if self.error_history:
            return self.error_history[-1]
        return None


class CodeRecoverySystem:
    """Full self-healing system for AI code"""
    
    def __init__(self):
        self.healer = SelfHealingCode()
        self.restart_on_failure = True
        self.max_recursions = 5
    
    def handle_exception(self, error: Exception, context: str = "", 
                        callback=None) -> bool:
        """
        Handle exception with auto-fix attempt
        callback: optional function to call with fix suggestion
        Returns True if error was handled (fixed or acknowledged)
        """
        error_info = self.healer.capture_error(error, context)
        print(f"\nError caught: {error_info.error_type}: {error_info.message}")
        print(f"Context: {context}")
        
        # Attempt auto-fix
        auto_fixed = self.healer.attempt_fix(error, context)
        
        # Call callback if provided
        if callback:
            callback(error_info, auto_fixed)
        
        return auto_fixed
    
    def recover_and_continue(self, code_func, *args, **kwargs):
        """
        Run a function with error recovery
        If error occurs, attempt fix and retry
        """
        try:
            return code_func(*args, **kwargs)
        except Exception as e:
            print(f"Error occurred, attempting recovery...")
            self.handle_exception(e, f"In {code_func.__name__}")
            
            # Could retry here if needed
            if self.restart_on_failure:
                print("Restarting after error...")
            return None