import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

names = ["grecilia"]
usernames = ["happy"]
passwords = ["XXX"]

hashed_password = Hasher(passwords).generate()

# Menghasilkan hash untuk kata sandi
#hashed_password = bcrypt.hashpw(passwords.encode(), bcrypt.gensalt())
#hashed_password_str = hashed_password.decode()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_password, file)

