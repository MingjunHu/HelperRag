#实例化数据库
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
#文本处理
# 1. 文本处理函数
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uuid
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from langchain_core.documents import Document
from markitdown import MarkItDown
from sentence_transformers import CrossEncoder
import torch
import os
import re

def question_maker(ask,prompt_path,llm_key,model_name,temperature):
#问题制造机器
    ask_prompt=getText_4_path(prompt_path)

    # os.environ["ZHIPUAI_API_KEY"] = llm_key
    chat = ChatZhipuAI(
        model=model_name,
        temperature=temperature,
        zhipuai_api_key=llm_key,
    )
    messages = [
        # AIMessage(content="Hi."),
        SystemMessage(content=ask_prompt),
        HumanMessage(content=ask),
    ]
    response = chat.invoke(messages)
    print("问题有："+response.content)  # Displays the AI-generated poem
    return response.content

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """
    将长文本分割成小的文本块。

    Args:
        text (str): 原始文本。
        chunk_size (int): 每个文本块的大小。
        chunk_overlap (int): 相邻文本块的重叠部分。

    Returns:
        list: 包含文本块的列表。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

#根据路径获取文本：
def getText_4_path(file_path):
    """
    读取文件内容 (使用 with open)。

    Args:
        file_path (str): 文件路径。

    Returns:
        str: 文件内容，如果文件不存在或读取失败，返回 None。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# 5. 将文本块和对应的向量存储到 Milvus
def save_to_milvus(text_chunks, mivus_vector,file_path):
    """
    将文本块和对应的向量存储到 Milvus。
    Args:
        text_chunks (list): 文本块列表
        embeddings (OpenAIEmbeddings): OpenAI embeddings 实例
    """
    ids = [str(uuid.uuid4()) for _ in text_chunks]
    file_name = os.path.basename(file_path) # Extract file name
    documents = [Document(page_content=text, metadata={"file_name": file_name, "file_path": file_path}) for text in text_chunks]

    mivus_vector.add_documents(documents, ids=ids)
    print(f"Inserted {len(text_chunks)} vector embeddings using langchain.")


    # vectors = embeddings.embed_documents(text_chunks)
    # data = [ids, text_chunks, vectors]

    # collection = Collection(vector_store)
    # collection.insert(data)
    # collection.flush()
    print(f"Inserted {len(text_chunks)} vector embeddings.")

def text_2_embedding(input_text,embeddings):
    vector = embeddings.embed_query(input_text)
    return vector

def get_all_file_paths_and_save_to_milvus(directory):
    """
    获取目录及其子目录中所有文件的绝对路径。

    Args:
        directory (str): 目标目录的路径。

    Returns:
        list: 包含所有文件绝对路径的列表。
    """

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(file_path)
            text=getText_4_path(file_path)
            chunk=chunk_text(text)
            save_to_milvus(chunk,vector_store,file_path)

    return file_paths


def remove_html_tags(html_content):
    """移除 HTML 标签."""
    clean_text = re.sub(r'<[^>]*?>', '', html_content)  # 正则匹配删除所有标签
    return clean_text

def movefiel_for_MakItDown(source_file_path, target_file_path):
    """
    读取源文件并处理内容，将内容写入目标文件后删除源文件。
    """
    try:
        # 读取源文件并处理内容
        md = MarkItDown()
        result = md.convert(source_file_path)

        # 将内容写入目标文件
        with open(target_file_path, 'w', encoding='utf-8') as tgt_file:
            tgt_file.write(remove_html_tags(result.text_content))

        print(f"Copied content to: {target_file_path}")

        # 删除源文件
        os.remove(source_file_path)
        print(f"Deleted source file: {source_file_path}")

    except Exception as e:
        print(f"Error processing file {source_file_path}: {e}")

def process_and_move_files(source_dir, target_dir):
    """
    获取源目录中的所有文件名和绝对路径，将内容写入目标目录的文件后删除源文件。
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    """
    try:
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)

        # 遍历源目录中的所有文件
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                # 获取源文件的绝对路径
                source_file_path = os.path.join(root, file_name)

                # 确定目标文件的路径
                target_file_path = os.path.join(target_dir, file_name)
                movefiel_for_MakItDown(source_file_path,target_file_path)
                

    except Exception as e:
        print(f"Error accessing directories: {e}")

# 使用示例
source_directory = "/opt/aiworkspase/lc/booksrc"
target_directory = "/opt/aiworkspase/lc/input"
process_and_move_files(source_directory, target_directory)


#执行重排模型
def reranker_maker(query, documents, n=5):
    #1:maidalun1020/bce-reranker-base_v1;2:BAAI/bge-reranker-v2-minicpm-layerwise;3:BAAI/bge-reranker-large
    model = CrossEncoder('maidalun1020/bce-reranker-base_v1')
    # 构建输入数据 (query, document) pairs
    pairs = [[query, doc] for doc in documents]

    # 模型推理，计算相关性分数
    scores = model.predict(pairs)

    # 打印排序结果
    sorted_docs_with_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    # for doc, score in sorted_docs_with_scores:
    #     print(f"Score: {score:.4f} - Document: {doc}")
    
    # 返回前n个排序后的文档
    return [doc for doc, _ in sorted_docs_with_scores[:n]]


def query_4_text(query_text,embeddings,DB_URI):
#根据问题获取内容切片和切片来源
    
    vector_store2 = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": DB_URI},
    )

    results1 = vector_store2.similarity_search(query=query_text,k=5)
    vector_q_text=text_2_embedding(query_text,embeddings)
    results2=vector_store2.similarity_search_by_vector(embedding=vector_q_text,k=5)

    results=results1+results2

    befor_answer=""
    file_paths={}
    documents=[]
    for doc in results:
        #print(f"* {doc.page_content} [{doc.metadata}]")
        befor_answer=befor_answer+doc.page_content+"\n"
        documents.append(doc.page_content)
        file_paths[doc.metadata['file_path']]=doc.metadata['file_name']
    return befor_answer,file_paths,documents


def answer_maker(first_ask,asks,embeddings,DB_URI):
#用问题找答案资料，返回问题串，回复资料串，源文件信息，所有文本切片
    ask_texts=""
    answer_texts=""
    total_files={}
    total_docs=[]
    for ask in asks:
        ask_texts=ask_texts+ask+";\n"
        answer_text,file_lists,documents=query_4_text(ask,embeddings,DB_URI)

        total_docs.extend(documents)

        answer_texts=answer_texts+answer_text+"\n"

        # print(ask+"    ||   "+answer_text)
        total_files=total_files|file_lists
    
    #模型模型做重排处理
    reranker_total_docs=reranker_maker(first_ask,total_docs,5)
    
    return ask_texts,answer_texts,total_files,reranker_total_docs

def the_end_answer(ask_texts,answer_texts,answer_prompt_path):
#在answer_maker之后用
    answer_prompt=getText_4_path(answer_prompt_path)
    messages = [
        # AIMessage(content="Hi."),
        SystemMessage(content=answer_prompt),
        HumanMessage(content="问题是：\n"+ask_texts+"\n\n回复材料如下\n"+answer_texts),
    ]
    response = chat.invoke(messages)
    print(response.content)  # Displays the AI-generated poem
    return response.content