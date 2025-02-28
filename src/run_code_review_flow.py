"""
Asynchronous code review flow using httpx and crewai.flow.

This script defines an asynchronous flow for code review, leveraging the httpx library for
asynchronous HTTP requests and the crewai.flow library for flow management.  It interacts with
a chatbot service to generate code and a code review service to analyze the generated code.
Logging is implemented using the `logging` module for detailed tracking of events and errors.

The flow consists of two main stages:

1. `generate_code`:  Asynchronously requests code from a chatbot service.
2. `create_review`: Asynchronously sends the generated code to a code review service for analysis.

Error handling is implemented using `try...except` blocks to catch `RequestException` and
`json.JSONDecodeError` exceptions during HTTP requests and JSON parsing.  Detailed error messages 
are logged using the configured logger.

The `kickoff` method initiates the flow, ensuring proper resource management with `async with 
self.client:`.

The `main` function instantiates the flow, runs the `kickoff` method, and logs the final result.
"""

import asyncio
import json
import logging

import httpx
from crewai.flow.flow import Flow, listen, start
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
# Set the log level to INFO
logger.setLevel(logging.INFO)
# Prevent duplicate logs
logger.propagate = False  
# Check if the logger already has handlers to prevent duplicate entries
if not logger.handlers:
    # Add a handler (e.g., to console) if one doesn't already exist.
    handler = logging.StreamHandler()  # Sends logs to the console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AsyncCodeReviewFlow(Flow):
    """
    Asynchronous workflow for generating and reviewing code snippets.
    
    This flow interacts with an external chatbot to generate Python code and then
    sends the generated code to an external code review service for analysis.
    """

    def __init__(self):
        """Initializes the AsyncCodeReviewFlow with an asynchronous HTTP client."""
        super().__init__()
        self.client = httpx.AsyncClient()

    @start()
    async def generate_code(self):
        """
        Requests a chatbot to generate an inefficient implementation of merge sort.
        
        Returns:
            str: The generated Python code as a string, or an error message if the request fails.
        """
        headers = {"Content-Type": "application/json"}
        chatbot_url = "http://localhost:80/chatbot/invoke"
        chatbot_data = {"message": """write a naive python code to implemet merge sort in a 
        inneficient way your response shoud have only python code, without any comment or text 
        before and after the code"""}

        chatbot_message = None
        try:
            chatbot_response = await self.client.post(url=chatbot_url, headers=headers, 
                json=chatbot_data)
            response_json = chatbot_response.json()
            chatbot_message = response_json.get("content", "No content field found")
            logger.info("#> chatbot_message: %s", chatbot_message)
        except RequestException as e:
            logger.error("Request error: %s", e)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)

        return chatbot_message

    @listen(generate_code)
    async def create_review(self, code):
        """
        Sends the generated code to a code review service for analysis.
        
        Args:
            code (str): The Python code to be reviewed.
        
        Returns:
            str: The review feedback from the code review service, or an error message if 
            the request fails.
        """
        headers = {"Content-Type": "application/json"}
        code_reviewer_url = "http://localhost:80/analyze-code"
        code_reviewer_data = {"message": code}
        logger.info("#> code_reviewer_data: %s", code_reviewer_data)

        code_reviewer_text = None
        try:
            code_reviewer_response = await self.client.post(url=code_reviewer_url, 
                headers=headers, json=code_reviewer_data)
            code_reviewer_text = code_reviewer_response.text
            logger.info("#> code_reviewer_response: %s", code_reviewer_response.text)
        except RequestException as e:
            logger.error("Request error: %s", e)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)

        return code_reviewer_text

    async def kickoff(self):
        """
        Initiates the asynchronous workflow by executing the flow.
        
        Returns:
            str: The final reviewed code feedback.
        """
        async with self.client:
            return await super().kickoff()


async def main():
    """Runs the AsyncCodeReviewFlow and logs the reviewed code output."""
    flow = AsyncCodeReviewFlow()
    result = await flow.kickoff()
    logger.info("#> Reviewed code: %s", result)

if __name__ == "__main__":
    asyncio.run(main())
