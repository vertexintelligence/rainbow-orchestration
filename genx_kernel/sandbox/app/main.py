from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import tempfile
import os

app = FastAPI(title="GenX Sandbox Runner v1.0")

class CodeRun(BaseModel):
    language: str  # "python" only for v1
    code: str

class RunResult(BaseModel):
    ok: bool
    stdout: str
    stderr: str
    exit_code: int

@app.post("/v1/run", response_model=RunResult)
def run_code(req: CodeRun):
    if req.language != "python":
        return RunResult(ok=False, stdout="", stderr="Only python allowed in v1.", exit_code=2)

    os.makedirs("/tmp/genx", exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp/genx", delete=False) as f:
        f.write(req.code)
        path = f.name

    try:
        p = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=10
        )
        ok = (p.returncode == 0)
        return RunResult(ok=ok, stdout=p.stdout, stderr=p.stderr, exit_code=p.returncode)
    except subprocess.TimeoutExpired:
        return RunResult(ok=False, stdout="", stderr="Timeout exceeded.", exit_code=124)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
