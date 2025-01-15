# backend/database.py
from pymilvus import Collection, connections

class Database:
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self.collection_name = "knowledge_base"
        self.connect()

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
    
    def get_vector_store(self,embeddings,DB_URI):
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": DB_URI},
        )

