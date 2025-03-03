#实例化数据库
# from langchain_milvus import Milvus
# from langchain_openai import OpenAIEmbeddings
#文本处理
# 1. 文本处理函数
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uuid
# from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from langchain_core.documents import Document
from markitdown import MarkItDown
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient
import torch
import os
import re
import pymysql,time
from utils.logger import LOG  # 导入日志记录模块

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
    LOG.debug("问题有："+response.content)  # Displays the AI-generated poem
    return response.content

def keywords_maker(ask,prompt_path,llm_key,model_name,temperature):
#关键词制造机器
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
    LOG.debug("关键词有："+response.content)  # Displays the AI-generated poem
    return response.content

def chunk_text(text, chunk_size=800, chunk_overlap=150):
    LOG.debug(f"[chunk_text]-----------------start-----------------")
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
    LOG.debug(f"[chunk_text]-----------------end----------------")
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
def save_to_milvus(content,text_chunks, Milvus_Client,DB_COLLECTION,file_path,bge_m3_ef):
    """
    将文本块和对应的向量存储到 Milvus。
    Args:
        text_chunks (list): 文本块列表
        embeddings (OpenAIEmbeddings): OpenAI embeddings 实例
    """
    LOG.debug(f"[save_to_milvus]-----------------start-----------------")
    try:
        datas=[]
        for text in text_chunks:
            data={}
            file_name = os.path.basename(file_path) # Extract file name
            data["text"]=remove_html_tags(text)
            data["dense_vector"]=bge_m3_ef.encode_documents([text])["dense"][0]
            data["content"]=remove_html_tags(text)
            data["id"]=file_path
            data["file_path"]=file_path
            data["clean_url"]=file_path
            data["file_name"]=file_name
            data["title"]=file_name

            # LOG.debug(data)
            datas.append(data)
        
        # print(uuids)
        LOG.debug(f"[save_to_milvus]-----------------add_documents-start-----------------")
        flag=Milvus_Client.insert(
            collection_name=DB_COLLECTION,
            data=datas
        )
        LOG.debug(f"[save_to_milvus]-----------------add_documents-end-----------------")
        # time.sleep(0.5)
    except Exception as e:
        LOG.info(f"Error show info accessing directories: {e}")
    LOG.info(f"[{file_path}] Inserted {len(text_chunks)} vector embeddings using Milvus.")

# 5. 将文本块和对应的向量存储到 Milvus
def save_to_milvus_old(text_chunks, mivus_vector,file_path):
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
    # print(f"Inserted {len(text_chunks)} vector embeddings.")

def save_to_milvus_sql_fields_old(text_chunks, mivus_vector,Milvus_Client,DB_COLLECTION,embeddings,file_path,sql_item_map):
    """
    将数库查出来的单条内容切割的文本块和对应的向量存储到 Milvus。
    Args:
        text_chunks (list): 文本块列表
        embeddings (OpenAIEmbeddings): OpenAI embeddings 实例
    """
    try:
        file_name = sql_item_map["title"] # Extract file name
        documents = [Document(
            page_content=remove_html_tags(text), 
            metadata={
                "file_name": file_name, 
                "file_path": file_path,
                "id":sql_item_map["id"],
                "tid":sql_item_map["tid"],
                "title":sql_item_map["title"],
                "nid":sql_item_map["nid"],
                "origin_id":sql_item_map["origin_id"],
                "co_name":sql_item_map["co_name"],
                "clean_url":sql_item_map["clean_url"],
                "content":remove_html_tags(sql_item_map["content"]),
                "author":sql_item_map["author"],
                "editor":sql_item_map["editor"],
                "founder_sender":sql_item_map["founder_sender"],
                "founder_auditor":sql_item_map["founder_auditor"],
                "copyfrom":sql_item_map["copyfrom"],
                "publish_time":str(sql_item_map["publish_time"])
            }
        ) for text in text_chunks]
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        # print(uuids)
            
        flag=mivus_vector.add_documents(documents, ids=uuids)
        time.sleep(0.5)
        # for doc in documents:
        #     print(doc.page_content)
        #     print("**********************************************")
        #     print(doc.metadata)
        #     print("##############################################")
    except Exception as e:
        print(f"Error show info accessing directories: {e}")
    
    print(f"[{str(sql_item_map['id'])}] Inserted {len(text_chunks)} vector embeddings using langchain.")

def save_to_milvus_sql_fields(text_chunks, mivus_vector,Milvus_Client,DB_COLLECTION,embeddings,bge_m3_ef,file_path,sql_item_map):
    """
    将数库查出来的单条内容切割的文本块和对应的向量存储到 Milvus。
    Args:
        text_chunks (list): 文本块列表
        embeddings (OpenAIEmbeddings): OpenAI embeddings 实例
    """
    LOG.debug(f"[save_to_milvus_sql_fields]-----------------start-----------------")
    try:
        datas=[]
        for text in text_chunks:
            data={}
            data["text"]=remove_html_tags(text)
            data["dense_vector"]=bge_m3_ef.encode_documents([text])["dense"][0]
            data["content"]=remove_html_tags(sql_item_map["content"])
            data["id"]=int(sql_item_map["id"])
            data["tid"]=int(sql_item_map["tid"])
            data["nid"]=int(sql_item_map["nid"])
            data["file_name"]=sql_item_map["title"]
            data["file_path"]=file_path
            data["title"]=sql_item_map["title"]
            data["clean_url"]=sql_item_map["clean_url"]
            data["author"]=sql_item_map["author"]
            data["editor"]=sql_item_map["editor"]
            data["founder_sender"]=sql_item_map["founder_sender"]
            data["founder_auditor"]=sql_item_map["founder_auditor"]
            data["publish_time"]=str(sql_item_map["publish_time"])

            # LOG.debug(data)
            datas.append(data)
        
        # print(uuids)
        LOG.debug(f"[save_to_milvus_sql_fields]-----------------add_documents-start-----------------")
        flag=Milvus_Client.insert(
            collection_name=DB_COLLECTION,
            data=datas
        )
        LOG.debug(f"[save_to_milvus_sql_fields]-----------------add_documents-end-----------------")
        # time.sleep(0.5)
    except Exception as e:
        LOG.info(f"Error show info accessing directories: {e}")
    LOG.info(f"[{str(sql_item_map['id'])}] Inserted {len(text_chunks)} vector embeddings using Milvus.")


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

def get_all_itmes_by_sql_and_save_to_milvus(sql,from_id_no,to_id_no,vector_store,Milvus_Client,DB_COLLECTION,embeddings,bge_m3_ef,
        host="127.0.0.1",
        user="root",
        password="@#c5y1o5l",
        db="spread",):
    try:
        LOG.debug(f"-----------------start-----------------")
        items_no=0
        vectory_items_no=0
        ids=[]
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
            cursorclass=pymysql.cursors.DictCursor
        )
        LOG.debug(f"-----------------connection-----------------")
        cursor = connection.cursor()
        if not to_id_no or to_id_no<=from_id_no:
            to_id_no=from_id_no+10
        # sql=sql+" where id >= "+str(from_id_no)+" and id <= "+str(to_id_no)
        LOG.debug("SQL_FOR_SELECT : "+sql)
        cursor.execute(sql)
        LOG.debug(f"-----------------execute-----------------")
        results = cursor.fetchall()
        LOG.debug(f"-----------------fetchall-----------------")
        cursor.close()
        connection.close()
        LOG.debug
        for result in results:
            LOG.debug
            temp_map={}
            temp_text=""
            for key, value in result.items():
                # print(f"{key}: {value}")
                temp_text+=key+":"+str(value)+"\n"
                temp_map[key]=value
            # chunk=chunk_text(remove_html_tags(temp_text))
            LOG.debug(f"-----------------temp_map-----------------")
            chunks=chunk_text(remove_html_tags("标题："+temp_map["title"]+"\n\n内容:"+temp_map["content"]))
            LOG.debug(f"-----------------chunk-----------------")
            save_to_milvus_sql_fields(chunks,vector_store,Milvus_Client,DB_COLLECTION,embeddings,bge_m3_ef,temp_map["clean_url"],sql_item_map=temp_map)
            items_no+=1
            vectory_items_no+=len(chunks)
            ids.append(temp_map["id"])
            LOG.debug(f"-----------------[id]:[{temp_map['id']}]-----------------")
        LOG.debug(f"-----------------end-----------------")
    except Exception as e:
        LOG.debug(f"Error accessing directories: {e}")
    return items_no,vectory_items_no,ids



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

        LOG.debug(f"Copied content to: {target_file_path}")

        # 删除源文件
        os.remove(source_file_path)
        print(f"Deleted source file: {source_file_path}")

    except Exception as e:
        LOG.debug(f"Error processing file {source_file_path}: {e}")

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
# source_directory = "/opt/aiworkspase/lc/booksrc"
# target_directory = "/opt/aiworkspase/lc/input"
# process_and_move_files(source_directory, target_directory)


#执行重排模型
def reranker_maker(query, documents, n=5):
    #1:maidalun1020/bce-reranker-base_v1;2:BAAI/bge-reranker-v2-minicpm-layerwise;3:BAAI/bge-reranker-large
    model = CrossEncoder('maidalun1020/bce-reranker-base_v1')
    # 构建输入数据 (query, document) pairs
    pairs = [[query, doc["content"]] for doc in documents]
    # 模型推理，计算相关性分数
    scores = model.predict(pairs)
    # 打印排序结果
    sorted_docs_with_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    # for doc, score in sorted_docs_with_scores:
    #     print(f"Score: {score:.4f} - Document: {doc}")
    
    # 返回前n个排序后的文档
    return [doc for doc, _ in sorted_docs_with_scores[:n]]

def query_by_vector_and_text(query_text,keywords,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id):
    documents=[]
    ids=[]
    key_array=[]
    LOG.debug("------------------------------start----------------------------")
    client = MilvusClient(uri=DB_URI,token=DB_TOKEN)
    try:
        #根据keywords按向量搜索
        query_vector_array=[]
        if query_text is not None and query_text != "": # 第一次进
            query_vector_array.append(bge_m3_ef.encode_documents([query_text])["dense"][0])
        LOG.debug(f"------------------------------{keywords}----------------------------")
        for keyword in keywords:
            if query_text is not None and query_text != "": # 第一次进
                if keyword in key_array:
                    continue
                else:
                    key_array.append(keyword)
                    query_vector_array.append(bge_m3_ef.encode_documents([keyword])["dense"][0])
            else:
                for k in keyword:
                    if k in key_array:
                        continue
                    else:
                        LOG.debug(f"------------------------------{k}----------------------------")
                        key_array.append(k)
                        query_vector_array.append(bge_m3_ef.encode_documents([k])["dense"][0])
        LOG.debug(f"[query_vector_array's length is:]-----------------{len(query_vector_array)}-----------------")
        if len(query_vector_array)>10:
            query_vector_array=query_vector_array[:10]
        res = client.search(
            collection_name=DB_COLLECTION,
            anns_field="dense_vector",
            data=query_vector_array,
            limit=10,
            search_params={"metric_type": "L2"},
            output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"]
        )
        for hits in res:
            LOG.debug("TopK results:")
            for doc in hits:
                if "0"==keowage_id:
                    LOG.debug(f"I AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    if doc["entity"]["id"] in ids:
                        LOG.debug(f"I BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
                        continue
                    else:
                        LOG.debug(f"I CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
                        ids.append(doc["entity"]["id"])
                else:
                    LOG.debug(f"I IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                    ids.append(doc["entity"]["id"])
                document={}
                document["id"]=doc["entity"]["id"]
                document["content"]=doc["entity"]["content"]
                document["title"]=doc["entity"]["title"]
                # LOG.debug(f"aaaaaa"+str(document["title"])+"-----------------{doc['entity']['title']}-----------------")
                document["clean_url"]=doc["entity"]["clean_url"]
                document["author"]=doc["entity"]["author"]
                document["editor"]=doc["entity"]["editor"]
                document["text"]=doc["entity"]["text"]
                documents.append(document)
    except Exception as e:
        print(f"根据keywords按稠密向量搜索 Error accessing directories: {e}")
    try:
        #根据全文搜索（稀疏向量）
        for keyword in key_array:
            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }
            res2=client.search(
                collection_name=DB_COLLECTION,
                data=[keyword],
                anns_field='sparse_vector',
                limit=10,
                output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
                search_params=search_params
            )
            for hits in res2:
                LOG.debug("TopK results:")
                for doc in hits:
                    if "0"==keowage_id:
                        LOG.debug(f"I JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ")
                        if doc["entity"]["id"] in ids:
                            continue
                        else:
                            ids.append(doc["entity"]["id"])
                    else:
                        LOG.debug(f"I KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
                        ids.append(doc["entity"]["id"])
                    document={}
                    document["id"]=doc["entity"]["id"]
                    document["content"]=doc["entity"]["content"]
                    document["title"]=doc["entity"]["title"]
                    # LOG.debug(f"bbbbbbb"+str(document["title"])+"-----------------{doc['entity']['title']}-----------------")
                    document["clean_url"]=doc["entity"]["clean_url"]
                    document["author"]=doc["entity"]["author"]
                    document["editor"]=doc["entity"]["editor"]
                    document["text"]=doc["entity"]["text"]
                    documents.append(document)
    except Exception as e:
        print(f"根据keywords按全文搜索 Error accessing directories: {e}")
    # try:
    #     #根据keywords按文本关键字搜
    #     title_str_keywords=""
    #     text_str_keywords=""
    #     for keyword in keywords:
            
    #         results_temp= client.query(
    #             collection_name=DB_COLLECTION,
    #             filter="(title like '%"+keyword+"%') or (text like '%"+keyword+"%')",
    #             output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
    #             limit=2
    #         )
    #         for doc in results_temp:
    #             if "0"==keowage_id:
    #                 if doc["id"] in ids:
    #                     continue
    #                 else:
    #                     ids.append(doc["id"])
    #             else:
    #                 ids.append(doc["id"])
    #             document={}
    #             document["id"]=doc["id"]
    #             document["content"]=doc["content"]
    #             document["title"]=doc["title"]

    #             # LOG.debug(f"ccccccc"+str(document["title"])+"-----------------{doc['title']}-----------------")

    #             document["clean_url"]=doc["clean_url"]
    #             document["author"]=doc["author"]
    #             document["editor"]=doc["editor"]
    #             document["text"]=doc["text"]
    #             documents.append(document)
    # except Exception as e:
    #     print(f"根据keywords按文本搜索 Error accessing directories: {e}")
    LOG.debug(f"-[documents   len]----------------{len(documents)}-----------------")
    return documents,ids

def query_by_doc_ids(main_ids,keywords,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id):
    documents=[]
    ids=[]
    key_array=[]
    try:
        client = MilvusClient(uri=DB_URI,token=DB_TOKEN)
        #根据keywords按向量搜索
        query_vector_array=[]
        for keyword in keywords:
            LOG.debug(f"------------------------------{keyword}----------------------------")
            for k in keyword:
                if k in key_array:
                    continue
                else:
                    LOG.debug(f"------------------------------{k}----------------------------")
                    key_array.append(k)
                    query_vector_array.append(bge_m3_ef.encode_documents([k])["dense"][0])
            # if keyword in key_array:
            #     continue
            # else:
            #     key_array.append(keyword)
            #     query_vector_array.append(bge_m3_ef.encode_documents([keyword])["dense"][0])
        LOG.debug(f"[query_vector_array's length is:]-----------------{len(query_vector_array)}-----------------")
        if len(query_vector_array)>10:
            query_vector_array=query_vector_array[:10]
        res1 = client.search(
            collection_name=DB_COLLECTION,
            anns_field="dense_vector",
            data=query_vector_array,
            filter = "id in {list}",
            filter_params = {"list": main_ids},
            search_params={"metric_type": "L2"},
            output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
            limit=10
        )
        LOG.debug(f"-----------------a-----------------")
        for hits in res1:
            for doc in hits:
                if "0"==keowage_id:
                    if doc["entity"]["id"] in ids:
                        continue
                    else:
                        ids.append(doc["entity"]["id"])
                else:
                    ids.append(doc["entity"]["id"])
                LOG.debug(f"[doc[entity][id]]-----------------{doc['entity']['id']}-----------------")
                document={}
                document["id"]=doc["entity"]["id"]
                document["content"]=doc["entity"]["content"]
                document["title"]=doc["entity"]["title"]
                
                # LOG.debug(f"dddddddd"+str(document["title"])+"-----------------{doc['entity']['title']}-----------------")
                document["clean_url"]=doc["entity"]["clean_url"]
                document["author"]=doc["entity"]["author"]
                document["editor"]=doc["entity"]["editor"]
                document["text"]=doc["entity"]["text"]
                documents.append(document)
        #根据全文搜索（稀疏向量）
        for keyword in key_array:
            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }
            res2=client.search(
                collection_name=DB_COLLECTION,
                data=[keyword],
                anns_field='sparse_vector',
                limit=10,
                output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
                search_params=search_params
            )
            for hits in res2:
                print("TopK results:")
                for doc in hits:
                    if "0"==keowage_id:
                        if doc["entity"]["id"] in ids:
                            continue
                        else:
                            ids.append(doc["entity"]["id"])
                    else:
                        ids.append(doc["entity"]["id"])
                    document={}
                    document["id"]=doc["entity"]["id"]
                    document["content"]=doc["entity"]["content"]
                    document["title"]=doc["entity"]["title"]
                    # LOG.debug(f"eeeeeeeee"+str(document["title"])+"-----------------{doc['entity']['title']}-----------------")
                    document["clean_url"]=doc["entity"]["clean_url"]
                    document["author"]=doc["entity"]["author"]
                    document["editor"]=doc["entity"]["editor"]
                    document["text"]=doc["entity"]["text"]
                    documents.append(document)

        LOG.debug(f"-----------------b-----------------")
        # #按关键字搜
        # for keyword in key_array:
        #     LOG.debug(f"[keyword]-----------------{keyword}-----------------")
        #     res2 = client.query(
        #         collection_name = DB_COLLECTION,
        #         filter = "(title like '%"+keyword+"%') or (text like '%"+keyword+"%')",
        #         output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
        #         limit = 10
        #     )
        #     LOG.debug(f"----------------------------------")
        #     for doc in res2:
        #         if "0"==keowage_id:
        #             if doc["id"] in ids:
        #                 continue
        #             else:
        #                 ids.append(doc["id"])
        #         else:
        #             ids.append(doc["id"])
        #         LOG.debug(f"doc[id]-----------------{doc['id']}-----------------")
        #         document={}
        #         document["id"]=doc["id"]
        #         document["content"]=doc["content"]
        #         document["title"]=doc["title"]
        #         # LOG.debug(f"fffffffff"+str(document["title"])+"-----------------{doc['entity']['title']}-----------------")
        #         document["clean_url"]=doc["clean_url"]
        #         document["author"]=doc["author"]
        #         document["editor"]=doc["editor"]
        #         document["text"]=doc["text"]
        #         documents.append(document)
    except Exception as e:
        print(f"Error accessing directories: {e}")
    return documents,ids


def query_4_text(query_text,ask_mainkeyword,ask_keywords_map,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id):
#根据问题获取内容切片和切片来源（方案二）
    
    
    LOG.debug(f"[query_4_text]-----------------start-----------------")
    # 1. 用户提出input query
    # 2. 大模型根据<input query>拆解成关键词<A><B><...>
    # 3. 大模型根据<input query>生成问题<gen question>
    # 4. 大模型根据<gen question>拆解成关键词<X><Y><...>
    LOG.debug(f"[query_4_text]-----------------4-----------------")
    ask_texts=""
    hit_keywords=[]
    ahit_documents=[]
    hit_ids=[]
    for ask,keyword in ask_keywords_map.items():
        ask_texts=ask_texts+ask+";\n"
        hit_keywords.append(keyword)
    # 5. 在向量和标量列中同时搜索<A> or <B> or <...>结果组成主键list<main_ids>
    LOG.debug(f"[query_4_text]-----------------5-----------------")
    main_documents,main_ids=query_by_vector_and_text(query_text,ask_mainkeyword,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id)
    # int_main_ids=[int(x) for x in main_ids]
    LOG.debug(f"[query_4_text]-----------------6-----------------")
    # 6. 在<main_ids>中的每个<id>的向量和标量列中搜索关键词<X> or <Y> or <...>
    # 7. 将搜到<X> or <Y> or <...>的结果保存为<hit_ids>
    LOG.debug(f"[query_4_text]-----------------7--------{len(main_documents)}---------")
    LOG.debug(f"keowage_id====:{keowage_id}")
    LOG.debug(f"hit_keywords length ====:{len(keowage_id)}")
    if("0"==keowage_id):
        LOG.debug("keowage=======o")
        ahit_documents,hit_ids=query_by_doc_ids(main_ids,hit_keywords[:10],embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id)
    else:
        LOG.debug("I'm here!111111111111111111111111")
        ahit_documents,hit_ids=query_by_vector_and_text(None,hit_keywords[:10],embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id)

    # 8. 将<main_ids>中的<hit_ids>删除并生成<nohit_ids>
    LOG.debug(f"[query_4_text]-----------------8--------{len(ahit_documents)}---------")
    if "0"==keowage_id:
        nohit_ids=[x for x in main_ids if x not in hit_ids]
        nohit_documents=[doc for doc in main_documents if doc["id"] not in nohit_ids]
    else:
        nohit_ids=main_ids
        nohit_documents=main_documents

    # 9. 将<nohit_ids>append到<hit_ids>结尾生成<output_ids>
    LOG.debug(f"[query_4_text]-----------------9----------{len(nohit_documents)}-------")
    output_ids=hit_ids+nohit_ids
    output_documents=ahit_documents+nohit_documents
    # output_documents=query_by_doc_ids(output_ids,DB_URI,DB_TOKEN,DB_COLLECTION)
    
    LOG.debug(f"[query_4_text]-----------------10----------{len(output_documents)}-------")

    # 10. 将<output_ids>中的文档按照向量和标量列中的匹配度排序
    # reranker_total_docs=reranker_maker(query_text,output_documents,5)
    return None,None,output_documents,output_ids,ask_texts



def query_4_text_old(query_text,ask_mainkeyword,keywords,embeddings,DB_URI,DB_TOKEN,DB_COLLECTION,ids):
#根据问题获取内容切片和切片来源
    
    befor_answer=""
    file_paths={}
    documents=[]
    metadatas=[]
    try:
        # vector_store2 = Milvus(
        #     embedding_function=embeddings,
        #     connection_args={"uri": DB_URI,"token": DB_TOKEN},
        #     # collection_name=DB_COLLECTION,
        # )

        # results1 = vector_store2.similarity_search(query=query_text,k=5)

        # for doc in results1:
        #     #判断是否重复id
        #     if doc.metadata["id"] in ids:
        #     # if any(value == doc.metadata["id"] for value in ids):
        #         continue
        #     else:
        #         ids.append(doc.metadata["id"])
        #     document={}
        #     document["id"]=doc.metadata["id"]
        #     document["content"]=doc.metadata["content"]
        #     document["title"]=doc.metadata["title"]
        #     document["clean_url"]=doc.metadata["clean_url"]
        #     document["author"]=doc.metadata["author"]
        #     document["editor"]=doc.metadata["editor"]
        #     document["text"]=doc.page_content
        #     documents.append(document)


        #     ####old###########
        #     # befor_answer=befor_answer+doc.page_content+"\n"
        #     # documents.append(doc.page_content)
        #     # # metadatas.append(doc.metadata)
        #     # file_paths[doc.metadata['file_path']]=doc.metadata['file_name']
        #     # # ids.append(doc.metadata['id'])
        
        client = MilvusClient(uri=DB_URI,token=DB_TOKEN)

        # query_vector_array=[]
        # for keyword in keywords:
        #     query_vector_array.append(text_2_embedding(keyword,embeddings))
        # # query_vector_array.append(text_2_embedding(ask_mainkeyword[0],embeddings))
        # query_vector_array.append(text_2_embedding(query_text,embeddings))        
        # # query_vector = text_2_embedding(query_text,embeddings)
        # # query_vector2 = text_2_embedding(ask_mainkeyword[0],embeddings)

        # res = client.search(
        #     collection_name=DB_COLLECTION,
        #     anns_field="vector",
        #     # data=[query_vector,query_vector2],
        #     # data=[query_vector],
        #     data=query_vector_array,
        #     limit=2,
        #     search_params={"metric_type": "L2"},
        #     output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"]
        # )
        # for hits in res:
        #     print("TopK results:")
        #     for doc in hits:
        #         # print(doc)
        #         print("=====================================")
        #         #判断是否重复id
        #         if doc["entity"]["id"] in ids:
        #         # if any(value == doc.metadata["id"] for value in ids):
        #             continue
        #         else:
        #             ids.append(doc["entity"]["id"])
        #         document={}
        #         document["id"]=doc["entity"]["id"]
        #         document["content"]=doc["entity"]["content"]
        #         document["title"]=doc["entity"]["title"]
        #         document["clean_url"]=doc["entity"]["clean_url"]
        #         document["author"]=doc["entity"]["author"]
        #         document["editor"]=doc["entity"]["editor"]
        #         document["text"]=doc["entity"]["text"]
        #         documents.append(document)



        print("1111111111111111111111111111111111111")
        
        title_str_keywords=""
        text_str_keywords=""
        for keyword in keywords:
            print(keyword)
            title_str_keywords=title_str_keywords+" title like '%"+keyword+"%' or"
            text_str_keywords=text_str_keywords+" text like '%"+keyword+"%' or"
        title_str_keywords=title_str_keywords[:-2]
        text_str_keywords=text_str_keywords[:-2]
        select_sql_where="(title like '%"+ask_mainkeyword[0]+"%' and ("+title_str_keywords+")) or (text like '%"+ask_mainkeyword[0]+"%' and ("+text_str_keywords+"))"

        print("title_str_keywords=============="+title_str_keywords)
        print("text_str_keywords--------------"+text_str_keywords)
        print("select_sql_where=============="+select_sql_where)
        
        results_temp= client.query(
            collection_name=DB_COLLECTION,
            filter=select_sql_where,
            output_fields=["id", "title", "content","text","clean_url","author","editor","founder_sender","founder_auditor","copyfrom","publish_time"],
            limit=2
        )
        
        for doc in results_temp:
            #判断是否重复id
            if doc["id"] in ids:
            # if any(value == doc["id"] for value in ids):
                continue
            else:
                ids.append(doc["id"])
            document={}
            document["id"]=doc["id"]
            document["content"]=doc["content"]
            document["title"]=doc["title"]
            document["clean_url"]=doc["clean_url"]
            document["author"]=doc["author"]
            document["editor"]=doc["editor"]
            document["text"]=doc["text"]
            documents.append(document)

            # #print(f"* {doc.page_content} [{doc.metadata}]")
            # befor_answer=befor_answer+doc["content"]
            # documents.append(doc["content"])
            # # metadatas.append(doc.metadata)
            # file_paths[doc["clean_url"]]=doc["title"]
            # # ids.append(doc["id"])
    except Exception as e:
        print(f"Error accessing directories: {e}")
    return befor_answer,file_paths,documents,ids


def answer_maker(first_ask,ask_mainkeyword,ask_keywords_map,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id):
#用问题找答案资料，返回问题串，回复资料串，源文件信息，所有文本切片
    ask_texts=""
    answer_texts=""
    total_files={}
    total_docs=[]
    use_ids=[]
    other_files={}

    #方案2
    answer_text,file_lists,total_docs,ids,ask_texts=query_4_text(first_ask,ask_mainkeyword,ask_keywords_map,embeddings,bge_m3_ef,DB_URI,DB_TOKEN,DB_COLLECTION,keowage_id)
    #方案1
    # for ask,keywords in ask_keywords_map.items():
    #     print("ids==============")
    #     print(ids)
    #     ask_texts=ask_texts+ask+";\n"
    #     answer_text,file_lists,documents,temp_ids=query_4_text_old(ask,ask_mainkeyword,keywords,embeddings,DB_URI,DB_TOKEN,DB_COLLECTION,ids)
        
    #     ids=temp_ids
    #     total_docs.extend(documents)

        # answer_texts=answer_texts+answer_text+"\n"

        # print(ask+"    ||   "+answer_text)
        # total_files=total_files|file_lists
    
    LOG.debug(f"[total_docs len]----------{len(total_docs)}----")
    #模型模型做重排处理
    # reranker_total_docs=reranker_maker(first_ask,total_docs,5)
    if keowage_id=="0":
        if(len(total_docs)>7):
            reranker_total_docs=total_docs[:7]
        else:
            reranker_total_docs=total_docs
    else:
        if(len(total_docs)>15):
            reranker_total_docs=total_docs[:15]
        else:
            reranker_total_docs=total_docs
    LOG.debug(f"[reranker_total_docs len]----------{len(reranker_total_docs)}----")
    for doc in reranker_total_docs:
        # LOG.debug(f"[拼接]content------{str(doc['title'])}----||----{str(doc['content'])}")
        answer_texts=answer_texts+doc["content"]+"\n"
        use_ids.append(doc["id"])
        total_files[doc["clean_url"]]=doc["title"]
    
    for o_doc in total_docs:
        if o_doc["id"] in use_ids:
            continue
        else:
            other_files[o_doc["clean_url"]]=o_doc["title"]
    LOG.debug(f"[other_files len]----------{len(other_files)}----")
    
    return ask_texts,answer_texts,total_files,reranker_total_docs

def the_end_answer(ask_texts,answer_texts,answer_prompt_path):
#在answer_maker之后用
    answer_prompt=getText_4_path(answer_prompt_path)
    messages = [
        # AIMessage(content="Hi."),
        SystemMessage(content=answer_prompt),
        HumanMessage(content="问题是：\n"+ask_texts+"\n\n回复材料如下：\n"+answer_texts),
    ]
    response = chat.invoke(messages)
    print(response.content)  # Displays the AI-generated poem
    return response.content