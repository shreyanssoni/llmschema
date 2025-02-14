import logging

class LLMValidationError(Exception):
    """Raised when the LLM response fails to match the expected schema."""
    
    def __init__(self, message: str, errors: list = None, response_json: dict = None):
        """
        Args:
            message (str): High-level error message.
            errors (list, optional): Specific validation errors (if any). Default is None.
            response_json (dict, optional): The raw LLM response that caused the failure. Default is None.
        """
        self.message = message
        self.errors = errors or []
        self.response_json = response_json

        # Log the error details
        logging.error(f"LLMValidationError: {message}")
        if self.errors:
            logging.error(f"Validation Errors: {self.errors}")
        if self.response_json:
            logging.error(f"Raw LLM Response: {self.response_json}")

        super().__init__(self.__str__())

    def __str__(self):
        """Custom error string representation."""
        error_details = "\n- " + "\n- ".join(self.errors) if self.errors else "No detailed errors provided."
        return f"{self.message}\nValidation Issues:{error_details}\nRaw Response: {self.response_json}"
