"""
This the bot with same functionality as the discord bot but with gradio instead of discord.
This is to show off the bot with less time and effort.
And this could be greatful while testing the bot.
"""

from pathlib import Path
import gradio as gr

from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig

# ChromaDB configuration
chroma_config = ChromaDBConfig(collection_name="aegion")

# SQLite configuration
sqlite_config = SQLiteConnectionConfig(db_file=Path("../aegion.db"), dataset_table_name="new_dataset")

# Gradio interface
with gr.Blocks(title="LUNA Chatbot") as demo:
    gr.Markdown("# LUNA Chatbot")

    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox(show_copy_button=True)

    clear = gr.ClearButton([msg, chatbot])

    btn = gr.Button("submit")

    nlqs_instance = NLQS(sqlite_config, chroma_config)

    msg.submit(nlqs_instance.execute_nlqs_workflow, [msg, chatbot], [msg, chatbot])
    btn.click(nlqs_instance.execute_nlqs_workflow, [msg, chatbot], [msg, chatbot])

try:
    demo.launch(debug=True, share=True)
except Exception as e:
    print(e)
    demo.launch(debug=True)
