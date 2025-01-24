import asyncio
import json
import requests

from requests.exceptions import RequestException
from crewai.flow.flow import Flow, listen, start

class CodeReviewFlow(Flow):

    @start()
    def generate_code(self):
        headers = {"Content-Type": "application/json"}
        chatbot_url = "http://localhost:80/chatbot/invoke"
        chatbot_data = {"message": """write a naive python code to implemet merge sort in a inneficient way. 
                your response shoud have only python code, without any comment or text before and after the code"""}

        try:
            chatbot_response = requests.post(chatbot_url, headers=headers, json=chatbot_data, verify=False)  # 'verify=False' is equivalent to '--insecure'
            response_json = chatbot_response.json()
            chatbot_message = response_json.get("content", "No content field found")
            print("#> chatbot_message:", chatbot_message)
        except RequestException as e:
            print(f"#> HTTP Request Error: {e}")
            return "Error, sorry!"
        except json.JSONDecodeError:
            print("#> Response is not in JSON format:", response_json.text)
            return "Error, sorry!"

        return chatbot_message

    @listen(generate_code)
    def create_review(self, chatbot_message):
        headers = {"Content-Type": "application/json"}
        code_reviewer_url = "http://localhost:80/analyze-code"
        code_reviewer_data = {"message": chatbot_message}
        print("#> code_reviewer_data:", code_reviewer_data)

        try:
            code_reviewer_response = requests.post(code_reviewer_url, headers=headers, json=code_reviewer_data, verify=False)  # 'verify=False' is equivalent to '--insecure'
            print("#> code_reviewer_response:", code_reviewer_response.text)    
        except RequestException as e:
            print(f"#> HTTP Request Error: {e}")
            return "Error, sorry!"
        except json.JSONDecodeError:
            print(f"#> Invalid JSON response: {code_reviewer_response.text}")
            return "Error, sorry!"
        
        return code_reviewer_response.text

async def main():
    flow = CodeReviewFlow()
    result = await flow.kickoff()
    print(f"#> Reviewed code: {result}")

asyncio.run(main())
