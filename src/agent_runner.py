from src.agents import llm_with_tools
from src.tools import search_tool, save_research_note
from src.logger import logger


# Available tools
tools = {
    "tavily_search": search_tool,
    "save_research_note": save_research_note,
}


def run_agent(user_request: str):

    messages = [
        {
            "role": "user",
            "content": user_request
        }
    ]

    while True:

        # Ask the LLM what to do next
        response = llm_with_tools.invoke(messages)

        print("\n===== MODEL RESPONSE =====")
        print(response)

        # Check if model requested tools
        if not response.tool_calls:
            return response.content


        # Execute requested tools
        if response.tool_calls:

            tool_call = response.tool_calls[0]

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"🔧 Executing tool: {tool_name}")

            tool_result = tools[tool_name].invoke(tool_args)
            logger.info(
    f"Executing {tool_name} with arguments {tool_args}"
)



            # Send tool result back to model
            messages.append(response)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result)
                }
            )