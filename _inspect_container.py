"""One-off script: inspect library versions inside the container."""
import chromadb
from pathlib import Path
import langchain_community, inspect

print("chromadb:", chromadb.__version__)
print("langchain_community:", langchain_community.__version__)

try:
    from langchain_community.chat_models import ChatOllama
    src = inspect.getsourcefile(ChatOllama)
    print("ChatOllama src:", src)
    text = Path(src).read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if "/api/" in stripped:
            print("  endpoint line:", stripped)
except Exception as e:
    print("ChatOllama inspect error:", e)

try:
    from chromadb.config import Settings
    s = Settings(anonymized_telemetry=False)
    print("Chroma Settings fields:", list(s.__fields__.keys()) if hasattr(s, '__fields__') else dir(s))
except Exception as e:
    print("Chroma Settings error:", e)

# Check telemetry implementation
try:
    import chromadb.telemetry
    from pathlib import Path as P
    tel_path = P(chromadb.telemetry.__file__).parent
    for f in sorted(tel_path.rglob("*.py")):
        txt = f.read_text(errors="ignore")
        if "capture(" in txt:
            print("telemetry file:", f)
            for i, ln in enumerate(txt.splitlines(), 1):
                if "capture(" in ln:
                    print(f"  line {i}: {ln.strip()}")
except Exception as e:
    print("Telemetry inspect error:", e)
