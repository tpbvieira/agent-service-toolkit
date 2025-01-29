import asyncio
import json
import logging

import httpx
from crewai.flow.flow import Flow, listen, start
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Set the log level to INFO
logger.setLevel(logging.INFO)

# Add a handler (e.g., to console) if one doesn't already exist.  
# This is crucial; otherwise, you won't see any log output.
handler = logging.StreamHandler()  # Sends logs to the console
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AsyncCodeReviewFlow(Flow):

    def __init__(self):
        super().__init__()
        self.client = httpx.AsyncClient()

    @start()
    async def generate_code(self):
        headers = {"Content-Type": "application/json"}
        chatbot_url = "http://localhost:80/chatbot/invoke"
        chatbot_data = {"message": """write a naive python code to implemet merge sort in a inneficient way. 
                your response shoud have only python code, without any comment or text before and after the code"""}
        
        chatbot_message = None
        try:
            chatbot_response = await self.client.post(url=chatbot_url, headers=headers, json=chatbot_data)
            response_json = chatbot_response.json()
            chatbot_message = response_json.get("content", "No content field found")
            logger.info("#> chatbot_message: %s", chatbot_message)
        except RequestException as e:
            logger.error(e)
        except json.JSONDecodeError as e:
            logger.error(e)

        return chatbot_message
    
    @listen(generate_code)
    async def create_review(self, code):
        headers = {"Content-Type": "application/json"}
        code_reviewer_url = "http://localhost:80/analyze-code"
        code_reviewer_data = {"message": code}
        logger.info("#> code_reviewer_data: %s", code_reviewer_data)

        code_reviewer_text = None
        try:
            code_reviewer_response = await self.client.post(url=code_reviewer_url, headers=headers, json=code_reviewer_data)
            code_reviewer_text = code_reviewer_response.text
            logger.info("#> code_reviewer_response: %s", code_reviewer_response.text)
        except RequestException as e:
            logger.error(e)
        except json.JSONDecodeError as e:
            logger.error(e)

        return code_reviewer_text
    
    async def kickoff(self):
        async with self.client:
            return await super().kickoff()


async def main():
    flow = AsyncCodeReviewFlow()
    result = await flow.kickoff()
    logger.info("#> Reviewed code: %s", result)

if __name__ == "__main__":
    asyncio.run(main())