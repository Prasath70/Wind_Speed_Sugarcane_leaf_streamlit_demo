import threading
import uvicorn
import subprocess

def start_fastapi():
    subprocess.run([
        "python", "-m","src.Ml_Server"
    ])

def start_streamlit():
    subprocess.run([
        "streamlit", "run", "src/Streamlitapp.py"
    ])

if __name__ == "__main__":
    threading.Thread(target=start_fastapi, daemon=True).start()
    threading.Thread(target=start_streamlit, daemon=True).start()

    print("\nSmart Agro AI running...")
    input("Press ENTER to stop.\n")
