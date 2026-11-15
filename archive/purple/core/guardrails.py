import re

class Guardrails:
    """
    Ensures the AI doesn't perform destructive or sensitive actions.
    """
    FORBIDDEN_COMMANDS = [
        r"rm\s+-rf\s+/",
        r"format\s+.",
        r"del\s+/s\s+/q",
        r"mkfs",
        r"chmod\s+-R\s+777",
        r"shutdown",
        r"reboot"
    ]
    
    SENSITIVE_KEYWORDS = [
        "bank", "password", "login", "credit card", "ssn", "social security"
    ]

    def is_safe(self, code):
        """Checks if a piece of code is safe to execute."""
        for pattern in self.FORBIDDEN_COMMANDS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Command matches forbidden pattern: {pattern}"
        
        return True, "Safe"

    def filter_response(self, text):
        """Redacts sensitive information from responses."""
        # Simplified redaction
        processed_text = text
        for keyword in self.SENSITIVE_KEYWORDS:
            processed_text = re.sub(keyword, "[REDACTED]", processed_text, flags=re.IGNORECASE)
        return processed_text
