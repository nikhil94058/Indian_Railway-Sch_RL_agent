Set the Python Path (Temporary for this session):
code
Powershell
$env:PYTHONPATH = "$PWD"
Run uvicorn using the full module path:
code
Powershell
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000 --reload
