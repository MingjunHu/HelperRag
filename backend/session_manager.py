#/backend/session_manager.py
import time
import uuid
from threading import Lock

class SessionData:
    def __init__(self, expiration_time):
        self.history = []  # 存储会话历史
        self.expiration_time = expiration_time  # 会话过期时间（秒）
        self.last_accessed = time.time()  # 上次访问时间

    def is_expired(self):
        return (time.time() - self.last_accessed) > self.expiration_time

    def update_access_time(self):
        self.last_accessed = time.time()

class SessionManager:
    def __init__(self, expiration_minutes=30):
        self.sessions = {}  # 存储会话数据
        self.lock = Lock()  # 线程安全锁
        self.expiration_time = expiration_minutes * 60  # 转换为秒

    def create_session(self):
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = SessionData(self.expiration_time)
        return session_id

    def get_history(self, session_id):
        with self.lock:
            session_data = self.sessions.get(session_id)
            if session_data:
                session_data.update_access_time()  # 更新访问时间
                return session_data.history
            return []

    def add_to_history(self, session_id, message):
        with self.lock:
            session_data = self.sessions.get(session_id)
            if session_data:
                session_data.history.append(message)
                session_data.update_access_time()  # 更新访问时间

    def cleanup_sessions(self):
        """定期清理过期的会话"""
        with self.lock:
            expired_sessions = [session_id for session_id, session_data in self.sessions.items() if session_data.is_expired()]
            for session_id in expired_sessions:
                del self.sessions[session_id]

    def set_expiration_time(self, expiration_minutes):
        self.expiration_time = expiration_minutes * 60  # 转换为秒