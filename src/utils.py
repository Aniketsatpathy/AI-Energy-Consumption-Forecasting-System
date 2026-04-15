import os

def create_dirs():
    """
    Create required folders
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)