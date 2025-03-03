import gradio as gr
import requests
import uuid
import json
import os
from datetime import datetime

# 后端 API 地址
BACKEND_API = "http://localhost:8001"

# 生成随机 session ID
def generate_session_id():
    return str(uuid.uuid4())

# 初始化 session ID
session_id = generate_session_id()

# CSS样式
css = """
.confirm-dialog {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000 !important;
    background: white !important;
    padding: 20px !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0,0,0,0.5) !important;
    width: 400px !important;
}
.confirm-dialog::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: -1;
}
.warning-text {
    color: #ff4444;
    font-weight: bold;
}
"""

# 发送消息到后端
def send_message(input_text):
    session_id = generate_session_id()
    print(f"Sending message: {input_text} with session_id: {session_id}")
    
    response = requests.post(
        BACKEND_API + "/chat",
        json={"input_text": input_text, "session_id": session_id},
    )
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("response", "没有返回内容")
    else:
        return f"Error: {response.status_code}, Message: {response.text}"

def validate_files(files):
    if not files:
        return False, "请选择文件"
    
    # 检查文件数量限制
    if len(files) > 100:
        return False, f"一次最多只能上传100个文件，当前选择了{len(files)}个文件"
    
    # 检查文件类型
    for file in files:
        if not file.name.lower().endswith('.txt'):
            return False, f"文件 {file.name} 不是txt格式，只允许上传txt文件"
    return True, ""

def submit_documents(files):
    if not files:
        return "请选择文件", None, None, None
    
    # 验证文件
    is_valid, error_message = validate_files(files)
    if not is_valid:
        return error_message, None, None, None
    
    # 上传文件
    uploaded_files = []
    for file in files:
        try:
            # 获取原始文件名
            original_filename = os.path.basename(file.name)
            files_dict = {"file": (original_filename, open(file.name, "rb"))}
            response = requests.post(f"{BACKEND_API}/upload_document", files=files_dict)
            
            if response.status_code == 200:
                uploaded_files.append(original_filename)
            else:
                return f"上传失败: {original_filename} - {response.text}", None, None, None
        except Exception as e:
            return f"上传出错: {original_filename} - {str(e)}", None, None, None
        finally:
            # 确保文件被关闭
            if 'files_dict' in locals() and 'file' in files_dict:
                files_dict['file'][1].close()
    
    # 刷新文件列表
    return f"成功上传以下文件:\n" + "\n".join(uploaded_files), get_file_list(), get_vectorized_list(), get_db_files()

def get_file_list():
    try:
        response = requests.get(f"{BACKEND_API}/list_documents")
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            
            # 格式化文件信息为表格数据
            formatted_files = []
            for file in files:
                size_kb = file['size'] / 1024
                modified_time = datetime.fromtimestamp(file['modified_time']).strftime('%Y-%m-%d %H:%M:%S')
                formatted_files.append([
                    file['filename'],
                    f"{size_kb:.2f} KB",
                    modified_time,
                    file['vectorization_status']  # 添加向量化状态
                ])
            
            return formatted_files
        else:
            return []
    except Exception as e:
        print(f"Error fetching file list: {str(e)}")
        return []

def get_vectorized_list():
    try:
        response = requests.get(f"{BACKEND_API}/list_vectorized_documents")
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            
            # 格式化文件信息为表格数据
            formatted_files = []
            for file in files:
                size_kb = file['size'] / 1024
                modified_time = datetime.fromtimestamp(file['modified_time']).strftime('%Y-%m-%d %H:%M:%S')
                formatted_files.append([
                    file['filename'],
                    f"{size_kb:.2f} KB",
                    modified_time
                ])
            
            return formatted_files
        else:
            return []
    except Exception as e:
        print(f"Error fetching vectorized file list: {str(e)}")
        return []

def get_db_files():
    try:
        response = requests.get(f"{BACKEND_API}/list_db_files")
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            
            # 格式化文件信息为表格数据
            formatted_files = []
            for file in files:
                size_kb = file['size'] / 1024
                # modified_time = datetime.fromtimestamp(file['modified_time']).strftime('%Y-%m-%d %H:%M:%S')
                formatted_files.append([
                    file['filename'],
                    f"{size_kb:.2f} KB",
                    # modified_time,
                    # file['type']
                ])
            
            return formatted_files
        else:
            return []
    except Exception as e:
        print(f"Error fetching database file list: {str(e)}")
        return []

def delete_all_files():
    try:
        response = requests.post(f"{BACKEND_API}/delete_all_documents")
        
        if response.status_code == 200:
            result = response.json()
            deleted_count = result.get('total_deleted', 0)
            message = f"成功删除所有文件，共 {deleted_count} 个文件"
            
            # 刷新文件列表
            return message, get_file_list(), get_vectorized_list(), get_db_files()
        else:
            return f"删除失败: {response.text}", None, None, None
    except Exception as e:
        return f"删除出错: {str(e)}", None, None, None

def delete_all_data():
    try:
        response = requests.post(f"{BACKEND_API}/delete_all_data")
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('message', '')
            
            # 刷新文件列表
            return message, get_file_list(), get_vectorized_list(), get_db_files()
        else:
            return f"删除失败: {response.text}", None, None, None
    except Exception as e:
        return f"删除出错: {str(e)}", None, None, None

def vectorize_documents():
    try:
        response = requests.post(f"{BACKEND_API}/vectorize_documents")
        
        if response.status_code == 200:
            result = response.json()
            processing_count = result.get('processing_count', 0)
            message = f"开始向量化处理，共 {processing_count} 个文件"
            
            # 刷新文件列表
            return message, get_file_list(), get_vectorized_list(), get_db_files()
        else:
            return f"向量化处理失败: {response.text}", None, None, None
    except Exception as e:
        return f"向量化处理出错: {str(e)}", None, None, None

def refresh_lists():
    return get_file_list(), get_vectorized_list(), get_db_files()

# Gradio 界面
with gr.Blocks(css=css) as demo:
    # 存储当前操作的状态
    current_files = gr.State(None)
    current_action = gr.State(None)
    
    # 确认对话框
    with gr.Group(visible=False, elem_classes="confirm-dialog") as confirm_dialog:
        gr.Markdown("### 确认操作")
        confirm_text = gr.Markdown()
        with gr.Row():
            confirm_yes = gr.Button("确认", variant="primary")
            confirm_no = gr.Button("取消")
    
    gr.Markdown("# 搜问小助手 v0.0.1版本")
    gr.Markdown("### 本工具提供以下功能：\n1. 问答搜索\n2. SQL数据库操作\n3. 数据操作")

    with gr.Tab("搜索问答"):
        chatbot = gr.Chatbot(type='messages')
        msg = gr.Textbox(placeholder="请输入你的问题...")
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history):
            user_message = history[-1]["content"]
            bot_response = send_message(user_message)
            print(f"Bot response: {bot_response}")
            history.append({"role": "assistant", "content": bot_response})
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    with gr.Tab("SQL数据库操作"):
        gr.Markdown("### SQL数据库操作")
        with gr.Row():
            sql_input = gr.Textbox(label="SQL查询语句", value="select * from app_article where id> 195225 and id<=201714 AND series='imagetext' and (content like '%清明节%' or content like '%五四%')",placeholder="输入SQL查询语句...")
        with gr.Row():
            from_id_no = gr.Textbox(label="起始id号", value=154468,placeholder="输入起始id号...")
            to_id_no = gr.Textbox(label="终止id号", value=154480,placeholder="输入终止id号...")
        gr.Markdown("### 数据库配置")
        with gr.Row():
            host_input = gr.Textbox(label="数据库地址", value="127.0.0.1")
            user_input = gr.Textbox(label="用户名", value="root")
            password_input = gr.Textbox(label="密码", type="password", value="@#c5y1o5l")
            db_input = gr.Textbox(label="数据库名", value="spread")
        gr.Markdown("### 结果")
        sql_output = gr.Textbox(label="操作结果")
        sql_result = gr.Dataframe(
            headers=["处理结果"],
            type="array",
            interactive=False
        )
        
        sql_button = gr.Button("执行SQL并保存", variant="primary")
        
        def handle_sql(sql_text, from_id_no,to_id_no,host, user, password, db):
            try:
                response = requests.post(
                    f"{BACKEND_API}/save_from_sql",
                    json={
                        "sql_text": sql_text,
                        "from_id_no": from_id_no,
                        "to_id_no": to_id_no,
                        "host": host,
                        "user": user,
                        "password": password,
                        "db": db
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    return f"成功处理 {result['items_no']} 条数据，已向量化 {result['vectory_items_no']} 条", [[f"ID: {id}"] for id in result['ids']]
                return f"错误: {response.text}", []
            except Exception as e:
                return f"操作失败: {str(e)}", []

        sql_button.click(
            handle_sql,
            inputs=[sql_input,from_id_no,to_id_no, host_input, user_input, password_input, db_input],
            outputs=[sql_output, sql_result]
        )


    with gr.Tab("数据操作"):
        gr.Markdown("### 文件上传（限制：最多100个txt文件）")
        with gr.Row():
            file_input = gr.File(file_count="multiple", file_types=[".txt"])
            upload_button = gr.Button("上传文件", variant="primary")
            refresh_button = gr.Button("刷新列表")
        
        output_message = gr.Textbox(label="操作结果")
        
        # 待处理文件列表
        gr.Markdown("### 待处理文件列表")
        file_list = gr.Dataframe(
            headers=["文件名", "大小", "修改时间", "向量化状态"],
            type="array",
            interactive=False,
            label="待处理文件列表"
        )
        
        with gr.Row():
            delete_button = gr.Button("删除待向量文件", variant="stop")
            vectorize_button = gr.Button("文件向量化入库", variant="primary")
            delete_data_button = gr.Button("清空数据库", variant="stop")
        
        # 已向量化文件列表
        gr.Markdown("### 已向量化文件列表")
        vectorized_list = gr.Dataframe(
            headers=["文件名", "大小", "修改时间"],
            type="array",
            interactive=False,
            label="已向量化文件列表"
        )
        
        # 数据库文件列表
        gr.Markdown("### 数据库文件列表")
        db_file_list = gr.Dataframe(
            headers=["文件名", "大小"],
            type="array",
            interactive=False,
            label="数据库文件列表"
        )
        
        # 事件处理
        def show_confirm(message, action, files=None):
            return {
                confirm_dialog: gr.update(visible=True),
                confirm_text: message,
                current_action: action,
                current_files: files
            }
        
        def hide_confirm():
            return {
                confirm_dialog: gr.update(visible=False),
                current_action: None,
                current_files: None
            }
        
        def handle_confirm(action, files):
            if action == "upload":
                return submit_documents(files)
            elif action == "delete":
                return delete_all_files()
            elif action == "vectorize":
                return vectorize_documents()
            elif action == "delete_data":
                return delete_all_data()
            return None, None, None, None
        
        # 上传文件确认流程
        upload_button.click(
            lambda x: show_confirm("确认要上传选中的文件吗？", "upload", x),
            inputs=[file_input],
            outputs=[confirm_dialog, confirm_text, current_action, current_files]
        )
        
        # 删除确认流程
        delete_button.click(
            lambda: show_confirm("确认要删除待向量化的文件吗？此操作不可恢复！", "delete"),
            outputs=[confirm_dialog, confirm_text, current_action, current_files]
        )
        
        # 向量化确认流程
        vectorize_button.click(
            lambda: show_confirm("确认要将文件向量化并入库吗？", "vectorize"),
            outputs=[confirm_dialog, confirm_text, current_action, current_files]
        )
        
        # 数据删除确认流程
        delete_data_button.click(
            lambda: show_confirm(
                """<div class='warning-text'>警告：此操作将清空数据库！</div>
                <br/>
                这将删除：
                1. backup目录下的所有文件
                2. 数据库目录下的所有文件
                <br/>
                此操作不可恢复！确定要继续吗？""",
                "delete_data"
            ),
            outputs=[confirm_dialog, confirm_text, current_action, current_files]
        )
        
        # 确认按钮处理
        confirm_yes.click(
            handle_confirm,
            inputs=[current_action, current_files],
            outputs=[output_message, file_list, vectorized_list, db_file_list]
        ).then(
            hide_confirm,
            outputs=[confirm_dialog, current_action, current_files]
        )
        
        # 取消按钮处理
        confirm_no.click(
            hide_confirm,
            outputs=[confirm_dialog, current_action, current_files]
        )
        
        # 刷新按钮
        refresh_button.click(
            refresh_lists,
            outputs=[file_list, vectorized_list, db_file_list]
        )

        # 页面加载时自动刷新文件列表
        demo.load(
            refresh_lists,
            outputs=[file_list, vectorized_list, db_file_list]
        )

demo.launch(server_name="0.0.0.0",server_port=8002,root_path="/kno")
