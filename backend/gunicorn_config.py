import os
import multiprocessing

# Bind to this socket
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")

# Number of worker processes
workers = 1  # Start with just one worker for debugging

# Worker class - use sync for easier debugging
worker_class = "sync"

# Timeout (seconds)
timeout = 120

# Log level
loglevel = "debug"

# Access log - writes to stdout by default
accesslog = "-"

# Error log - writes to stderr by default
errorlog = "-"

# Whether to send flask output to the error log
capture_output = True

# Path to a file where the application should store its PID
pidfile = "gunicorn.pid"

# Whether to daemonize the Gunicorn process (run in background)
daemon = False

# Restart workers after this many requests
max_requests = 1000

# Restart workers after this many seconds
max_requests_jitter = 30

# Pre-fork hooks (runs before forking)
def pre_fork(server, worker):
    server.log.info("Pre-fork hook: about to fork worker")

# Post-fork hooks (runs after forking)
def post_fork(server, worker):
    server.log.info(f"Post-fork hook: worker {worker.pid} forked")

# When a worker starts
def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

# When a worker exits
def worker_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exited")

# When a worker gets killed
def worker_abort(worker):
    worker.log.info(f"Worker {worker.pid} was aborted") 