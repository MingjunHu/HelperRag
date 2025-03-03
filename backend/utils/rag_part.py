# res1 = client.search(
#     collection_name=DB_COLLECTION,
#     anns_field = "vector",
#     data = #【【vector of <X> or <Y> or <...> 】】,
#     # 这里如果有多个次要关键词，可以搜索多次，也可以用 batch search
#     # 但是 batch search 需要再重组多次结果，建议这里先按一个关键词查找，后期再调整
#     search_params = {"metric_type": "L2"},
#     filter = "id in {list}",
#     filter_params = {"list": main_ids},
#     limit = 3,
#     # 这里的 limit 可以设置 3-5 个，如果太多则可能语义相似度不够高
# )

# # 将 res1 中的 id 存入 <em_hit_ids>，这里省略实现

# keywords = [X, Y]
# for keyword in keywords:
#     res2 = client.query(
#         collection_name = DB_COLLECTION,
#         filter = "(title like '%"+keyword+"%') or (text like '%"+keyword+"%')",
#         limit = 3,
#     )
#     # 将 res2 中的 id 追加到 <em_hit_ids> 中并去重，这里默认以向量结果优先
#     # 如果希望以标量结果优先，颠倒 search 和 query 的执行顺序即可