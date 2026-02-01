import os
import gradio as gr
from groq import Groq
from datetime import datetime
import logging

# =========================
# CONFIGURATION
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure API key is set in Colab:
# os.environ["GROQ_API_KEY"] = "your_key_here"  (ONLY for testing, not GitHub)

# =========================
# CHATBOT CLASS
# =========================

class AICareerChatbot:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(" GROQ_API_KEY not found. Set it as an environment variable.")

        self.client = Groq(api_key=self.groq_api_key)

        self.models = {
            "Fast Model (LLaMA 3.1 8B)": "llama-3.1-8b-instant",
            "Advanced Model (LLaMA 3.3 70B)": "llama-3.3-70b-versatile"
        }

        self.conversation_history = []

    def create_career_prompt(self, user_question):
        system_prompt = """
You are an expert AI Career Guidance Counselor.
Format response with:
- Career Overview
- Required Skills & Technologies
- Learning Roadmap
- Recommended Resources
- Salary Expectations
- Next Steps
"""
        context = ""
        if self.conversation_history:
            context = "\nConversation Context:\n"
            for entry in self.conversation_history[-3:]:
                context += f"User: {entry['user']}\nAssistant: {entry['assistant'][:150]}...\n\n"

        return f"{system_prompt}\n{context}\nUser Question: {user_question}"

    def get_career_response(self, question, model_choice, temperature):
        try:
            model_name = self.models[model_choice]
            prompt = self.create_career_prompt(question)

            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI Career Guidance Counselor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1024
            )

            response = completion.choices[0].message.content

            self.conversation_history.append({
                "user": question,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })

            self.conversation_history = self.conversation_history[-10:]
            return response

        except Exception as e:
            logger.error(str(e))
            return f" Error: {e}"

    def create_interface(self):
        with gr.Blocks(title="AI Career Guidance Chatbot") as demo:
            gr.Markdown("# 🚀 AI Career Guidance Chatbot")

            question = gr.Textbox(label="Your Question", lines=3)
            model = gr.Dropdown(choices=list(self.models.keys()),
                                value="Fast Model (LLaMA 3.1 8B)",
                                label="Model")
            temp = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Creativity")

            output = gr.Textbox(lines=18, label="Response")
            status = gr.Textbox(value="Ready", interactive=False)

            btn = gr.Button("Get Career Guidance ")

            btn.click(
                self.get_career_response,
                inputs=[question, model, temp],
                outputs=output
            )

        return demo

    def launch(self):
        self.create_interface().launch(share=True)


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    chatbot = AICareerChatbot()
    chatbot.launch()