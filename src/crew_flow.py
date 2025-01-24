import asyncio
from crewai.flow.flow import Flow, listen, start
from litellm import completion

class BlogContentFlow(Flow):
    model = "gpt-4"

    @start()
    def generate_topic(self):
        topic = "tópico gerado pelo llm"
        print(f"Generated Topic: {topic}")
        return topic

    @listen(generate_topic)
    def create_outline(self, topic):
        outline = "outine gerado pelo llm"
        print(outline)
        return outline

async def main():
    flow = BlogContentFlow()
    result = await flow.kickoff()
    print(f"Final Outline: {result}")

asyncio.run(main())