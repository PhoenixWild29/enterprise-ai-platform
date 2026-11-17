import os

# Tests run on in-process stand-ins; set before any module reads the environment.
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")
os.environ.setdefault("IDENTITY_PROVIDER", "simulated")
os.environ.setdefault("AGENT_BACKEND", "simulated")
