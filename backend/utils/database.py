# backend/database.py
from pymilvus import Collection, connections
from pymilvus import MilvusClient, DataType,Function, FunctionType
class Database:
    def __init__(self,URI=None,TOKEN=None,COLLECTION=None, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self.collection_name = "knowledge_base"
        self.MilvusClient=self.get_Milvus_Collection(URI,TOKEN,COLLECTION)
        # self.connect()

    def connect(self):
        connections.connect("default", host=self.host, port=self.port)
        self.create_collection()

    def create_collection(self):
        # 定义字段
        if not Collection.exists(self.collection_name):
            # 定义 schema
            from pymilvus import FieldSchema, CollectionSchema, DataType

            id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
            vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # OpenAI embedding size
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)

            schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="Knowledge base collection")
            self.collection = Collection(name=self.collection_name, schema=schema)

    def save_document(self, document, embedding):
        # 将文档和其向量存储到 Milvus
        ids = self.collection.insert([[None], [embedding], [document]])
        return ids
    
    def get_Milvus_Collection(self,URI,TOKEN,COLLECTION="m3_BM25"):

        # URI="https://in03-03b10f9a5f9af87.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
        # TOKEN="5fd643af12519881d095da352c301a9d5da65b8e7f6a8d8818d79afa3c9114a4d1d5d892bce2dafde1dbf09f644ab52ce4a54477"

        client = MilvusClient(
            uri=URI,
            token=TOKEN
        )

        # res = client.get_load_state(
        #     collection_name=COLLECTION
        # )
        # if res["state"] =='NotExist':
        #     print("collection No have:")


        #     #创建 schema
        #     schema = MilvusClient.create_schema()
        #     DIM = 1024

        #     schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True),
        #     #稀疏向量字段类型DataType.SPARSE_FLOAT_VECTOR，稠密向量datatype=DataType.FLOAT_VECTOR
        #     schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR), 
        #     schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
        #     schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="id", datatype=DataType.INT64),
        #     schema.add_field(field_name="tid", datatype=DataType.INT64),
        #     schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="nid", datatype=DataType.INT64),
        #     schema.add_field(field_name="origin_id", datatype=DataType.INT64),
        #     schema.add_field(field_name="clean_url", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="editor", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="founder_sender", datatype=DataType.VARCHAR, max_length=65535)
        #     schema.add_field(field_name="founder_auditor", datatype=DataType.VARCHAR, max_length=65535)
        #     schema.add_field(field_name="copyfrom", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="publish_time", datatype=DataType.VARCHAR, max_length=65535),
        #     schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000,enable_analyzer=True)

        #     bm25_function = Function(
        #         name="text_bm25_emb", # Function name
        #         input_field_names=["text"], # Name of the VARCHAR field containing raw text data
        #         output_field_names=["sparse_vector"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        #         function_type=FunctionType.BM25,
        #     )

        #     schema.add_function(bm25_function)

        #     schema.verify()

        #     #创建索引
        #     index_params = client.prepare_index_params()

        #     index_params.add_index(
        #         field_name="dense_vector",
        #         index_name="dense_index",
        #         index_type="IVF_FLAT",
        #         metric_type="L2",
        #         params={"nlist": 128},
        #     )

        #     index_params.add_index(
        #         field_name="sparse_vector",
        #         index_type="AUTOINDEX",
        #         metric_type="BM25",
        #     )

        #     index_params.add_index(
        #         field_name="id",
        #         index_type="STL_SORT"
        #     )

        #     #创建Collection
        #     client.create_collection(
        #         collection_name=COLLECTION,
        #         schema=schema,
        #         index_params=index_params
        #     )

        return client

