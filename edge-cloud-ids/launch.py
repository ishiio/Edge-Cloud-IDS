"""Set up an isolated environment, start the gateway, and open the dashboard."""
from pathlib import Path
import os, subprocess, sys, time, urllib.request, webbrowser

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
PYTHON = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def main():
    os.chdir(ROOT)
    if not PYTHON.exists():
        print('Creating your local Python environment...', flush=True)
        subprocess.check_call([sys.executable, '-m', 'venv', str(VENV)])
    requirements = (ROOT / 'requirements.txt').read_text()
    marker = VENV / '.ids-requirements'
    if not marker.exists() or marker.read_text() != requirements:
        print('Installing required packages (first launch only)...', flush=True)
        subprocess.check_call([str(PYTHON), '-m', 'pip', 'install', '-r', 'requirements.txt'])
        marker.write_text(requirements)
    print('Checking bundled models before starting the server...', flush=True)
    subprocess.check_call([str(PYTHON), '-c',
        'from service.engine import Bundle, available_datasets; '
        'keys = available_datasets(); '
        'assert keys, "No trained models found. Extract the complete ZIP."; '
        '[Bundle.load(key) for key in keys]; '
        'print("Model check passed.", flush=True)'], cwd=ROOT)
    print('Starting Edge-Cloud IDS. Leave this terminal open.', flush=True)
    child = subprocess.Popen([str(PYTHON), '-m', 'service.run'], cwd=ROOT)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError('The server stopped. Read the error above; port 8000 may already be in use.')
            try:
                with opener.open('http://127.0.0.1:8000/api/state', timeout=2) as r:
                    if r.status == 200:
                        webbrowser.open('http://127.0.0.1:8000/#datasets')
                        print('Ready: http://127.0.0.1:8000', flush=True)
                        break
            except Exception:
                time.sleep(.5)
        else:
            raise RuntimeError('Server startup timed out. Read the terminal output above.')
        child.wait()
    finally:
        child.terminate()
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired: child.kill()

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: pass
    except Exception as exc:
        print(f'\nStartup failed: {exc}\nYou can still open OPEN_DASHBOARD.html to browse the saved datasets.', flush=True)
        input('Press Enter to close...')
        sys.exit(1)
