# HelperRag

HelperRag 是一个基于向量数据库的知识库管理系统，支持文档的批量上传、向量化处理和检索功能。

## 功能特点

- 文件管理
  - 支持批量上传txt文件（最多100个）
  - 文件类型验证
  - 实时显示文件状态
  - 一键删除功能

- 向量化处理
  - 自动文本分块
  - 向量化存储
  - 状态实时显示
  - 处理进度跟踪

- 用户界面
  - 直观的文件列表显示
  - 操作确认弹窗
  - 实时状态更新
  - 响应式布局

## 技术栈

- 前端：Gradio
- 后端：Flask
- 向量数据库：Milvus
- 文本处理：LangChain

## 目录结构

```
HelperRag/
├── backend/                 # 后端服务
│   ├── agents/             # 智能代理模块
│   │   ├── agent_base.py
│   │   ├── knowage_agent.py
│   │   └── session_history.py
│   ├── logs/               # 日志目录
│   │   └── app.log
│   ├── prompts/            # 提示词模板
│   │   ├── knowage_answer_prompt.txt
│   │   └── knowage_query_prompt.txt
│   ├── utils/              # 工具类
│   │   ├── comon.py
│   │   ├── database.py
│   │   ├── llm.py
│   │   └── logger.py
│   ├── config.py           # 后端配置
│   ├── document_processor.py
│   ├── model_handler.py
│   ├── server.py           # 主服务器
│   └── session_manager.py
├── frontend/               # 前端应用
│   ├── app.py             # 主界面
│   ├── document_uploader.py
│   ├── session_manager.py
│   ├── util.py
│   └── vector_storage.py
├── config/                # 配置文件目录
│   └── config.yaml       # 主配置文件
├── backup/               # 已处理文件备份
├── db/                   # 数据库文件
│   └── milvus/          # Milvus数据目录
├── input/               # 文件处理中转目录
├── uploads/             # 文件上传目录
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明
```

## 安装说明

1. 克隆项目
```bash
git clone https://github.com/your-username/HelperRag.git
cd HelperRag
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置
- 复制 `config/config.yaml.example` 到 `config/config.yaml`
- 修改配置文件中的相关设置

4. 启动服务
```bash
# 启动后端服务
cd backend
python server.py

# 启动前端服务
cd frontend
python app.py
```

## 使用说明

1. 文件上传
   - 点击"上传文件"按钮选择txt文件
   - 确认上传
   - 查看上传状态

2. 向量化处理
   - 选择需要处理的文件
   - 点击"向量化知识库"按钮
   - 等待处理完成

3. 文件管理
   - 查看文件列表
   - 监控处理状态
   - 删除不需要的文件

## 版本历史

### v1.0.0
- 初始版本发布
- 基础文件管理功能
- 向量化处理支持
- 用户界面实现

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License

## 作者

Your Name

## 致谢

感谢所有贡献者的付出。
