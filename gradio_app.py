import gradio as gr
import requests
from src.logger import logger


# Function that calls FastAPI
def research_agent_ui(message, history, image):
    logger.info(f"Gradio UI sending request for: {message}")

    # For now we're not using the image
    if image is not None:
        logger.info("Image uploaded successfully.")

    api_url = "http://127.0.0.1:8000/research"
    payload = {
        "topic": message,
        "max_analysts": 3
    }

    try:
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            data = response.json()
            results = "\n\n".join(data["data"])
            return results

        return f"❌ API Error: {response.text}"

    except Exception as e:
        return f"❌ Connection Error: Ensure your FastAPI server is running! ({e})"


# Layout
with gr.Blocks(title="Rapid Agent Researcher") as demo:

    gr.Markdown("# 🚀 Rapid Agent Researcher")
    gr.Markdown(
        "Ask a topic and get concurrent expert analysis via Gradio."
    )

    with gr.Row():

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("## 📁 Multimodal Inputs")

            image_input = gr.Image(
                label="Upload Image",
                type="filepath"
            )

        # Main Chat Area
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=lambda msg, hist: research_agent_ui(
                    msg,
                    hist,
                    image_input.value
                ),
                examples=[
                    "Quantum Computing",
                    "DeepSeek R1 Architecture",
                    "Sustainable Energy"
                ]
            )

if __name__ == "__main__":
    demo.launch()
                