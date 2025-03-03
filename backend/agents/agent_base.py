import json
from abc import ABC, abstractmethod
import ast
# from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_ollama import OllamaEmbeddings  # 导入 OllamaEmbeddings 模型
from langchain_community.chat_models import ChatZhipuAI
from config import Config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
# from langchain_openai import OpenAIEmbeddings
# from langchain_milvus import Milvus

from .session_history import get_session_history  # 导入会话历史相关方法
from utils.logger import LOG  # 导入日志工具
import utils.comon as comon

from langchain_openai import OpenAI,ChatOpenAI  # 导入OpenAI库用于访问GPT模型

class AgentBase(ABC):
    """
    抽象基类，提供代理的共有功能。
    """
    def __init__(self, name, prompt_file,keywords_prompt_file,answer_prompt_file, intro_file=None, session_id=None):
        self.name = name
        self.prompt_file = prompt_file
        self.answer_prompt_file=answer_prompt_file
        self.keywords_prompt_file=keywords_prompt_file
        self.intro_file = intro_file
        self.session_id = session_id if session_id else self.name
        self.prompt = self.load_prompt()
        self.answer_prompt=self.load_answer_prompt()
        self.keywords_prompt=self.load_keywords_prompt()
        self.intro_messages = self.load_intro() if self.intro_file else []
        self.create_chatbot()

    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")
    def load_answer_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.answer_prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.answer_prompt_file}!")
    def load_keywords_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.keywords_prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.keywords_prompt_file}!")

    def load_intro(self):
        """
        从 JSON 文件加载初始消息。
        """
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到初始消息文件 {self.intro_file}!")
        except json.JSONDecodeError:
            raise ValueError(f"初始消息文件 {self.intro_file} 包含无效的 JSON!")

    def create_chatbot(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        self.config = Config()
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        answer_system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.answer_prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        keywords_system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.keywords_prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 初始化 ChatOllama 模型，配置参数
        # self.chatbot = system_prompt | ChatOllama(
        #     model="llama3.2:3b-instruct-q8_0",  # 使用的模型名称
        #     max_tokens=64192,  # 最大生成的 token 数
        #     temperature=0.7,  # 随机性配置
        # )

        # #初始化智普ai大模型
        # self.chatbot = system_prompt | ChatZhipuAI(
        #     zhipuai_api_key=self.config.get('model').get("api_key"),
        #     model=self.config.get('model').get("name"),
        #     max_tokens=self.config.get("model").get("parameters").get('max_tokens'),
        #     temperature=self.config.get("model").get("parameters").get('temperature')
        # )

        # #实例化问答模型（用于最终综合回答问题）
        # self.answer_chatboot=answer_system_prompt | ChatZhipuAI(
        #     zhipuai_api_key=self.config.get('model').get("api_key"),
        #     model=self.config.get('model').get("name"),
        #     max_tokens=self.config.get("model").get("parameters").get('max_tokens'),
        #     temperature=self.config.get("model").get("parameters").get('temperature')
        # )

        #Deepseek模型
        self.mymodel= ChatOpenAI(
            model="deepseek-chat",
            temperature=0.5,
            api_key="",
            base_url="https://api.deepseek.com/v1"
        )

        # self.mymodel= ChatZhipuAI(
        #     zhipuai_api_key=self.config.get('model').get("api_key"),
        #     model=self.config.get('model').get("name"),
        #     max_tokens=self.config.get("model").get("parameters").get('max_tokens'),
        #     temperature=self.config.get("model").get("parameters").get('temperature')
        # )

        self.chatbot = system_prompt | self.mymodel

        self.answer_chatboot=answer_system_prompt | self.mymodel

        #实例化OpenAI的向量模型Embedding
        # self.embeddings=OpenAIEmbeddings(
        #     model=self.config.get('openai').get("embedding"),
        #     #openai_api_base=self.config.get('openai').get("api_base"),
        #     openai_api_key=self.config.get('openai').get("api_key"),
        # )

        #实例化ollama嵌入模型Embedding
        # self.embeddings=OllamaEmbeddings(
        #     model=self.config.get('openai').get("embedding")
        # )
        self.embeddings=None

        self.bge_m3_ef = BGEM3EmbeddingFunction(
            model_name='/opt/aiworkspase/huggingface/BAAI/bge-m3', # Specify the model name
            device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
        )

        


        # #实例化免费向量模型，硅基流动
        # self.embeddings=OpenAIEmbeddings(
        #     model="BAAI/bge-m3",
        #     openai_api_base="https://api.siliconflow.cn/v1",
        #     openai_api_key="sk-wawayxqegimfwrasuohehdfkdyhucldnkndfvbzmytzsivrl",
        # )

        #实例化数据库
        # self.vector_store = Milvus(
        #     embedding_function=self.embeddings,
        #     connection_args={"uri": self.config.get("database").get("DB_URI"),"token":self.config.get("database").get("DB_TOKEN")},
        #     collection_name=self.config.get("database").get("DB_COLLECTION"),
        # )
        # self.vector_store =None

        # 将聊天机器人与消息历史记录关联
        self.chatbot_with_history = RunnableWithMessageHistory(self.answer_chatboot, get_session_history)

    def chat_with_history_v2(self, user_input,context, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。

        参数:
            user_input (str): 用户的问题
            content (str): 数据库查询的内容
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id
        content = content="问题是：\n"+user_input+"\n\n回复材料如下:\n"+context
        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=content)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )
        return response.content


    def chat_with_history(self, user_input, session_id=None,keowage_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id
        
        asks=ast.literal_eval(
            comon.question_maker(
                ask=user_input,
                prompt_path=self.prompt_file,
                llm_key=self.config.get('model').get("api_key"),
                model_name=self.config.get('model').get("name"),
                temperature=self.config.get('model').get("parameters").get("temperature")
            )
        )

        ask_mainkeyword=ast.literal_eval(
                    comon.keywords_maker(
                    ask=user_input,
                    prompt_path=self.keywords_prompt_file,
                    llm_key=self.config.get('model').get("api_key"),
                    model_name=self.config.get('model').get("name"),
                    temperature=self.config.get('model').get("parameters").get("temperature")
                )
            )

        ask_keywords_map={}
        for ask in asks:
            words=ast.literal_eval(
                    comon.keywords_maker(
                    ask=ask,
                    prompt_path=self.keywords_prompt_file,
                    llm_key=self.config.get('model').get("api_key"),
                    model_name=self.config.get('model').get("name"),
                    temperature=self.config.get('model').get("parameters").get("temperature")
                )
            )
            ask_keywords_map[ask]=words

        ask_texts,answer_texts,total_files,total_docs=comon.answer_maker(user_input,ask_mainkeyword,ask_keywords_map,self.embeddings,self.bge_m3_ef,self.config.get("database").get("DB_URI"),self.config.get("database").get("DB_TOKEN"),self.config.get("database").get("DB_COLLECTION"),keowage_id)

        #使用重排模型---暂时不用
        #comon.reranker_maker(ask_input,total_docs)
        print("--------------------------------------")

        # result_text=comon.the_end_answer(ask_texts,answer_texts,self.answer_prompt_file)
        
        print("------------------1111111111111--------------------")
        if keowage_id :
            print("i am kenwage")
            # content = content="问题是：\n"+ask_texts+"\n\n回复材料如下\n"+answer_texts
            content ="问题是：\n"+user_input+"\n\n回复材料如下：\n"+answer_texts
            response = self.chatbot_with_history.invoke(
                [HumanMessage(content=content)],  # 将用户输入封装为 HumanMessage
                {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
            )
        else:
            response = self.chatbot_with_history.invoke(
                [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
                {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
            )
        
        print("------------------22222222222222222222--------------------")
        # sorted_key=sorted(total_files.keys())

        
        print("-----------------333333333333333333333333---------------------")


        file_str="\n引用文件有："
        LOG.debug("------------aaaaaaaaaaaa-----------------")
        for doc in total_docs:
            LOG.debug("\n"+"<a href='"+doc["clean_url"]+"'>"+doc["title"]+"</a>")
            file_str+="\n"+"<a href='"+doc["clean_url"]+"'>"+doc["title"]+"</a>"

        # for key in total_files:
        #     LOG.debug("------------bbbbbbbbbbbbbb-----------------")
        #     # print(f"{key}   |   {filelists[key]}")
        #     if(key is None):
        #         continue
        #     else:
        #         if(total_files[key] is None):
        #             continue
        #     LOG.debug("\n"+"<a href='"+key+"'>"+total_files[key]+"</a>")
        #     file_str+="\n"+"<a href='"+key+"'>"+total_files[key]+"</a>"

        print("-----------------44444444444444444444444---------------------")

        # file_str+="\n\n你可能感兴趣文件："
        # for key in other_files:
        #     # print(f"{key}   |   {filelists[key]}")
        #     file_str+="\n"+"<a href='"+key+"'>"+total_files[key]+"</a>"

        print("-----------------5555555555555555555555---------------------")

        return response.content+file_str
        # return ast.literal_eval(response.content)