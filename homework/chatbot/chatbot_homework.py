import gradio as gr
import random, time

from typing import List
from enum import Enum, unique, auto

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

ENABLE_CHAT = False

# todo：电器、家装、教育
@unique
class SceneEnum(Enum):
  房产 = "real_estate"
  iPhone = "iphone"
  英语培训 = "english_training"

def initialize_sales_bot(vector_store_dir: str = "real_estate_sales"):
  db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  global SALES_BOT
  SALES_BOT = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(
      search_type="similarity_score_threshold",
      search_kwargs={"score_threshold": 0.8},
    ),
  )

  SALES_BOT.return_source_documents = True

  return SALES_BOT

def add_text(history, text):
  print(f"[history]{history}")
  print(f"[text]{text}")
  history = history + [(text, None)]
  return history, gr.update(value="", interactive=False)

# todo: change function name to bot_chat
def bot(history, text):
  query = history[-1][0]
  ans = SALES_BOT({"query": query})

  response = "这个问题我要问问领导"

  if  ans["source_documents"] or ENABLE_CHAT:
    print(f"[result]{ans['result']}")
    print(f"[source_documents]{ans['source_documents']}")
    response = ans["result"]

  history[-1][1] = ""

  for character in response:
    history[-1][1] += character
    time.sleep(0.05)
    yield history

def change_scene(choice):
  print(f"change_scene-[chlice]{choice}")
  vector_store_dir = choice + "_sales"
  initialize_sales_bot(vector_store_dir)
  return gr.Chatbot.update(value="")

def change_enable_chat(enable):
  global ENABLE_CHAT
  ENABLE_CHAT = enable

def launch_gradio_by_blocks():
  with gr.Blocks(title="销售机器人") as blocks:
    with gr.Row():
      with gr.Column(scale=1):
        with gr.Row():
          scene_radio = gr.Radio(
            [(member.name, member.value) for member in SceneEnum],
            label="切换场景",
            info="选择一个要咨询的场景",
            value=SceneEnum.房产
          )
          enable_chat_checkbox = gr.Checkbox(
            label="激活 AI",
            info="通过 AI 更好的回答问题",
            value=ENABLE_CHAT,
          )
      with gr.Column(scale=4):
        chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_with=False)
        with gr.Row():
          txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder=" 请输入你想咨询的问题",
            container=False,
          )
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=True).then(bot, chatbot, chatbot)
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=True)
    scene_radio.change(fn=change_scene, inputs=scene_radio, outputs=chatbot)
    enable_chat_checkbox.change(fn=change_enable_chat, inputs=enable_chat_checkbox)

  blocks.queue(max_size=10)
  blocks.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
  initialize_sales_bot()
  launch_gradio_by_blocks()