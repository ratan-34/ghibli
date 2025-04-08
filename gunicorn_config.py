import os
import multiprocessing

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
backlog = 2048

# Worker processes (optimized for Render's free tier)
workers = min(4, multiprocessing.cpu_count() * 2 + 1)
worker_class = 'gthread'  # Better for I/O bound apps than sync
threads = 2  # For worker_class 'gthread'
timeout = 120
keepalive = 2

# Process naming
proc_name = 'ghibli_style_transfer'

# Server mechanics
daemon = False
pidfile = None
umask = 0

# Logging (Render captures stdout/stderr)
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process management (helps prevent memory leaks)
max_requests = 500
max_requests_jitter = 50

# Timeouts (important for Render's free tier)
graceful_timeout = 30
worker_abort = 60

# Server hooks
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def when_ready(server):
    server.log.info(f"Server ready on {bind} with {workers} workers")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")
