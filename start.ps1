$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 run_local.py
} else {
    python run_local.py
}
