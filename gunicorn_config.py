# Gunicorn configuration for Render
import os
import multiprocessing

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
backlog = 2048

# Worker processes - Render free tier has limited resources
workers = 1  # For free plan, keep to 1
worker_class = 'sync'  # Use sync worker for better stability
worker_connections = 1000
timeout = 180  # Increased timeout for image processing operations
keepalive = 5

# Process naming
proc_name = 'gunicorn_animegan'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Request processing
max_requests = 500  # Lower for better memory management
max_requests_jitter = 50
graceful_timeout = 60  # Time to finish processing current requests before worker restart

# Thread configuration
threads = 2  # Number of threads per worker
thread_name_prefix = 'gunicorn_thread'

# Server hooks
def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")

def on_exit(server):
    server.log.info("Server is shutting down")

def worker_exit(server, worker):
    # Clean up after worker exits
    import gc
    gc.collect()
    server.log.info(f"Worker {worker.pid} exited, cleanup completed")
