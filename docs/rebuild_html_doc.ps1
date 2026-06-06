.  ../InvokeFunc.ps1
Invoke-NativeCommand uv run sphinx-build -M clean source/ _build/ -W --keep-going
Invoke-NativeCommand uv run sphinx-build -M html source/ _build/ -W --keep-going
