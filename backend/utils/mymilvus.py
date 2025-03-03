from pymilvus import MilvusClient, DataType,Function, FunctionType
class MyMilvus:
    def __init__(self,URI=None,TOKEN=None,COLLECTION=None):
        self.MilvusClient=self.get_Milvus_Collection(URI,TOKEN,COLLECTION)
        self.client=self.get_Milvus_Collection(URI=URI,TOKEN=TOKEN,COLLECTION=COLLECTION)
    
    def get_Milvus_Collection(self,URI,TOKEN,COLLECTION):

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