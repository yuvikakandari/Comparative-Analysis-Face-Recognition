from cryptography.fernet import Fernet
import os

KEY_FILE = "secret.key"

def load_or_create_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    with open(KEY_FILE, "rb") as f:
        return f.read()

key = load_or_create_key()
fernet = Fernet(key)

def save_encrypted(file_path, data):
    """Encrypts raw bytes and saves to a file."""
    encrypted_data = fernet.encrypt(data)
    with open(file_path, "wb") as f:
        f.write(encrypted_data)

def load_encrypted(file_path):
    """Reads an encrypted file and returns decrypted bytes."""
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    return fernet.decrypt(encrypted_data)