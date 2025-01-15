# backend/model_handler.py
import openai
import os
from agents.knowage_agent import KnowageAgent
from utils.logger import LOG
from langchain.chains import LLMChain
from langchain.llms import OpenAI as LangChainOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from config import Config
# from lc.helper.HelperRag.backend.utils.database import Database
from session_manager import SessionManager  # 导入 SessionManager

knowage_agent=KnowageAgent()

class ModelHandler:
    def __init__(self):
        self.config = Config()
        self.llm = ChatZhipuAI(
            zhipuai_api_key=self.config.get('model').get("api_key"),
            model=self.config.get('model').get("name"),
            max_tokens=self.config.get("model").get("parameters").get('max_tokens'),
            temperature=self.config.get("model").get("parameters").get('temperature')
        )
        #self.database = Database()
        
        # 从配置中读取过期时间
        expiration_time = self.config.get('session_manager').get('expiration_time_minutes', 5)
        self.session_manager = SessionManager(expiration_minutes=expiration_time)  # 创建 SessionManager 实例
        
        self.prompt_template = PromptTemplate(
            input_variables=["input_text", "history"],
            template="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\n{history}\n\nHuman: {input_text}\nAI:"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        # self.chain=self.prompt_template | self.llm

    # 处理用户输入的单词学习消息，并与词汇代理互动获取机器人的响应
    def handle_knowage(self,input_text, session_id,keowage_id="1"):
        bot_message = knowage_agent.chat_with_history(user_input=input_text,session_id=session_id,keowage_id="1")  # 获取机器回复
        LOG.info(f"[Knowage ChatBot]: {bot_message}")  # 记录机器人回应信息
        return bot_message


    def handle_chat(self, input_text, session_id,keowage_id="1"):
        # 获取历史记录
        # history = self.get_history(session_id)
        # 生成模型响应
        response = self.handle_knowage(input_text=input_text,session_id=session_id,keowage_id="1")
        # 更新历史记录
        # self.update_history(session_id, input_text, response)
        return response

    def get_history(self, session_id):
        return self.session_manager.get_history(session_id=session_id)

    def update_history(self, session_id, input_text, response):
        # 使用 session_manager 更新历史记录
        self.session_manager.add_to_history(session_id, f"Human: {input_text}")
        self.session_manager.add_to_history(session_id, f"AI: {response}")

    def embed_and_store(self, document):
        # 使用 OpenAI 生成嵌入
        response = openai.Embedding.create(
            input=document,
            model="text-embedding-ada-002"  # 更新为 text-embedding-ada-002
        )
        embedding = response['data'][0]['embedding']
        self.database.save_document(document, embedding)
