"""
Pytest configuration: set required environment variables before any app module
is imported so that pydantic-settings does not raise a ValidationError during
test collection.
"""
import os

# Dummy key so Settings() can be instantiated without a real .env
os.environ.setdefault("GROQ_API_KEY", "gsk-test-dummy-key-for-pytest")
