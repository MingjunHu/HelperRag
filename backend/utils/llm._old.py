#实例化数据库
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings


class LLM:
    def __init__(self,model_name="text-embedding-3-small",openai_api_key=None,openai_api_base=None):
        self.embeddings=OpenAIEmbeddings(
            model=model_name,
            #openai_api_base=
            openai_api_key=openai_api_key,
    )

