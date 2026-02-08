from langchain_openai import ChatOpenAI

def run_LLM(prompt: str) -> str:
    API_KEY = "sk-cafd1c5b525347a0b5bc44a55b30d3a0"
    BASE_URL = "https://api.deepseek.com"

    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1,
            max_tokens=2000
        )

        response = llm.invoke(prompt)

        # LangChain 返回的是 AIMessage
        if hasattr(response, "content"):
            return response.content
        else:
            return "no useful message return"

    except Exception as e:
        print(f"error: {str(e)}")
        return "system error, please try later."
