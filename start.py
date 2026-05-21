"""
=============================================================================
BLAZE DOUBLE AI — LAUNCHER v3.2
Inicia coletor + analisador + otimizador em paralelo com um comando só.
Uso: python start.py
=============================================================================
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

from analisador import load_sequence
from pantheon_engine import run_pantheon as run_ensemble, pantheon_init, pantheon_learn
from pantheon_engine import detect_micro_regime as detect_regime

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    cmd = [PYTHON, "-X", "utf8", os.path.join(BASE, script)]
    cmd.extend(args)
    
    return subprocess.Popen(
        cmd,
        cwd=BASE,
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
    )


def get_brave_browser():
    if os.name == "nt":
        candidates = []
        for root in (os.getenv("PROGRAMFILES"), os.getenv("PROGRAMFILES(X86)"), os.getenv("LOCALAPPDATA")):
            if root:
                candidates.append(os.path.join(root, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"))
        for path in candidates:
            if path and os.path.exists(path):
                return webbrowser.BackgroundBrowser(path)
    else:
        try:
            return webbrowser.get("brave")
        except webbrowser.Error:
            pass
    return None


def main():
    print("=" * 60)
    print(" BLAZE DOUBLE AI v3.2 — INICIANDO SISTEMA")
    print("=" * 60)
    print(f" Python     : {PYTHON}")
    print(f" Base       : {BASE}")
    print(f" Coletor    : {os.path.join(BASE, 'coletor.py')}")
    print(f" Analisador : {os.path.join(BASE, 'analisador.py')}")
    print(f" Otimizador : {os.path.join(BASE, 'otimizador_estrategias.py')}")
    print("=" * 60)
    print()

    print(" [1/3] Iniciando coletor... (porta 8765)")
    col = run("coletor.py")
    time.sleep(3)

    print(" [2/3] Iniciando analisador...")
    ana = run("analisador.py")
    time.sleep(2)

    print(" [3/3] Iniciando otimizador de estratégias em background...")
    oti = run("otimizador_estrategias.py", "--watch")
    time.sleep(2)

    print()
    print(" [4/4] Abrindo interfaces no navegador Brave...")
    brave = get_brave_browser()
    dashboard_uri = Path(BASE, "blaze-dashboard.html").as_uri()
    simulador_uri = Path(BASE, "simulador.html").as_uri()
    if brave:
        brave.open(dashboard_uri)
        time.sleep(1)
        brave.open(simulador_uri)
    else:
        print("[AVISO] Brave não encontrado; abrindo no navegador padrão.")
        webbrowser.open(dashboard_uri)
        time.sleep(1)
        webbrowser.open(simulador_uri)

    print()
    print(" Sistema rodando!")
    print(" API: http://localhost:8765/stats")
    print("      http://localhost:8765/analysis")
    print("      http://localhost:8765/signal")
    print()
    print(" Pressione Ctrl+C para encerrar tudo.")
    print("=" * 60)

    pantheon_init(colors=load_sequence(5000))

    try:
        while True:
            if col.poll() is not None:
                print("[AVISO] Coletor parou — reiniciando...")
                col = run("coletor.py")

            if ana.poll() is not None:
                print("[AVISO] Analisador parou — reiniciando...")
                ana = run("analisador.py")
                
            if oti.poll() is not None:
                print("[AVISO] Otimizador parou — reiniciando...")
                oti = run("otimizador_estrategias.py", "--watch")

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n Encerrando sistema...")
        try:
            col.terminate()
        except Exception:
            pass
        try:
            ana.terminate()
        except Exception:
            pass
        try:
            oti.terminate()
        except Exception:
            pass
        print(" Finalizado.")


if __name__ == "__main__":
    main()