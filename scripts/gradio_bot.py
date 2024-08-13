"""
This the bot with same functionality as the discord bot but with gradio instead of discord.
This is to show off the bot with less time and effort.
And this could be greatful while testing the bot.
"""

import gradio as gr

from nlqs.nlqs import NLQS
from nlqs.parameters import chroma_config, connection_config

# Gradio interface
with gr.Blocks(title="Chatbot using OpenAI") as demo:
    gr.Markdown("# Chatbot using OpenAI")

    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox(show_copy_button=True)

    clear = gr.ClearButton([msg, chatbot])

    btn = gr.Button("submit")

    nlqs_instance = NLQS(connection_config, chroma_config)

    msg.submit(nlqs_instance.execute_nlqs_workflow, [msg, chatbot], [msg, chatbot])
    btn.click(nlqs_instance.execute_nlqs_workflow, [msg, chatbot], [msg, chatbot])

demo.launch(debug=True, share=True)
