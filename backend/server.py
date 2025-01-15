import threading
import time
import os
import shutil
import utils.comon as comon
from flask import Flask, request, jsonify
from document_processor import DocumentProcessor
from model_handler import ModelHandler
from agents.knowage_agent import KnowageAgent

app = Flask(__name__)
document_processor = DocumentProcessor()
model_handler = ModelHandler()
knowageagent=KnowageAgent()
from langchain_milvus import Milvus

# 配置上传文件保存目录（使用绝对路径）
UPLOAD_FOLDER = model_handler.config.get("folder").get("uploads")
INPUT_FOLDER = model_handler.config.get("folder").get("input")
BACKUP_FOLDER = model_handler.config.get("folder").get("backup")

# 创建必要的目录
for folder in [UPLOAD_FOLDER, INPUT_FOLDER,BACKUP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
print(f"Upload directory: {UPLOAD_FOLDER}")
print(f"Input directory: {INPUT_FOLDER}")
print(f"Backup directory: {BACKUP_FOLDER}")

# 文件上传限制
MAX_FILES = 100

# 向量化状态
VECTORIZATION_STATUS = {
    "NOT_VECTORIZED": "未向量化",
    "PROCESSING": "向量化ing",
    "VECTORIZED": "已向量化"
}

# 存储文件向量化状态的字典
file_vectorization_status = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('input_text')
    session_id = data.get('session_id')
    print("input_text:"+input_text+"|session_id:"+session_id)

    if not input_text or not session_id:
        return jsonify({"error": "input_text and session_id are required"}), 400

    try:
        # 处理用户输入并获取模型响应
        response = model_handler.handle_chat(input_text, session_id,keowage_id="1")
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    # 获取当前上传目录中的文件数量
    current_files = len(os.listdir(UPLOAD_FOLDER))
    if current_files >= MAX_FILES:
        return jsonify({"error": f"Upload folder already has {current_files} files. Maximum limit is {MAX_FILES}"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # 验证文件类型
    if not file.filename.lower().endswith('.txt'):
        return jsonify({"error": "Only .txt files are allowed"}), 400
    
    try:
        # 构建目标文件路径
        target_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file.filename))
        
        # 保存文件到指定目录
        file.save(target_path)
        print(f"File uploaded successfully: {file.filename} to {target_path}")
        
        # 初始化文件的向量化状态
        file_vectorization_status[file.filename] = VECTORIZATION_STATUS["NOT_VECTORIZED"]
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": file.filename,
            "path": target_path
        }), 200
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/list_documents', methods=['GET'])
def list_documents():
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                # 如果文件不在状态字典中，初始化为未向量化
                if filename not in file_vectorization_status:
                    file_vectorization_status[filename] = VECTORIZATION_STATUS["NOT_VECTORIZED"]
                
                file_info = {
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "modified_time": os.path.getmtime(file_path),
                    "vectorization_status": file_vectorization_status[filename]
                }
                files.append(file_info)
        return jsonify({
            "files": files,
            "total": len(files)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_vectorized_documents', methods=['GET'])
def list_vectorized_documents():
    try:
        files = []
        for filename in os.listdir(BACKUP_FOLDER):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(BACKUP_FOLDER, filename)
                file_info = {
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "modified_time": os.path.getmtime(file_path)
                }
                files.append(file_info)
        return jsonify({
            "files": files,
            "total": len(files)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_all_documents', methods=['POST'])
def delete_all_documents():
    try:
        deleted_count = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                os.remove(file_path)
                # 清除向量化状态
                if filename in file_vectorization_status:
                    del file_vectorization_status[filename]
                deleted_count += 1
        
        return jsonify({
            "message": f"Successfully deleted all {deleted_count} files",
            "total_deleted": deleted_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/vectorize_documents', methods=['POST'])
def vectorize_documents():
    try:
        processing_count = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith('.txt'):
                # 只处理未向量化的文件
                if file_vectorization_status.get(filename) == VECTORIZATION_STATUS["NOT_VECTORIZED"]:
                    file_vectorization_status[filename] = VECTORIZATION_STATUS["PROCESSING"]
                    processing_count += 1
                    
                    # TODO: 启动异步任务处理向量化
                    # 1. 读取文件内容
                    # 2. 文本分块
                    # 3. 调用向量化模型
                    # 4. 存储到向量数据库
                    # 5. 更新状态为已向量化
                    source_directory = model_handler.config.get("folder").get("uploads")
                    target_directory = model_handler.config.get("folder").get("input")#UPLOAD_FOLDER+"/../input"
                    #备份目录
                    backup_directory = model_handler.config.get("folder").get("backup")#UPLOAD_FOLDER+"/../backup"
                    #上传文件路径
                    source_file_path = os.path.join(source_directory, filename)
                    #目标文件路径
                    target_file_path = os.path.join(target_directory, filename)
                    #备份文件路径
                    backup_file_path = os.path.join(backup_directory, filename)
                    #预处理文件，使用markitdown，移动文件
                    comon.movefiel_for_MakItDown(source_file_path, target_file_path)


                    try:
                        text=comon.getText_4_path(target_file_path)
                        chunk=comon.chunk_text(text)
                        vector_store = Milvus(
                            embedding_function=knowageagent.embeddings,
                            connection_args={"uri": model_handler.config.get("database").get("DB_URI")},
                        )
                        comon.save_to_milvus(chunk,vector_store,target_file_path)
                        comon.movefiel_for_MakItDown(target_file_path, backup_file_path)
                        # 这里临时模拟处理完成
                        file_vectorization_status[filename] = VECTORIZATION_STATUS["VECTORIZED"]

                    except Exception as e:
                        print(f"Error processing file {source_file_path}: {e}")
                    
        
        return jsonify({
            "message": f"Started vectorization for {processing_count} files",
            "processing_count": processing_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def cleanup_sessions():
    while True:
        time.sleep(300)  # 每 5 分钟清理一次过期会话
        model_handler.session_manager.cleanup_sessions()

if __name__ == "__main__":
    # 启动清理线程
    threading.Thread(target=cleanup_sessions, daemon=True).start()
    app.run(debug=True, port=5001)
