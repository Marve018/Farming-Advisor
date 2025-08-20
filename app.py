import gradio as gr

def farming_chat(message, history):

    return f"you said: {message}"

iface = gr.ChatInterface(
    fn=farming_chat,
    title="Igbo Language Farming Advisor",
    description="Ask questions about farming in English or Igbo language and get expert advice.",
    theme="soft"
)


if __name__ == "__main__":
    iface.launch()
