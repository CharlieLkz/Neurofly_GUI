import subprocess
import sys
import os
import time

def run_program_in_new_terminal(program_name):
    """
    Ejecuta un programa en una nueva terminal.
    """
    print(f"\n{'='*50}")
    print(f"Iniciando {program_name}...")
    print(f"{'='*50}\n")
    
    try:
        if sys.platform.startswith('win'):
            # Para Windows, usar 'start' para abrir nueva terminal
            cmd = f'start cmd /k python {program_name}'
            subprocess.Popen(cmd, shell=True)
        else:
            # Para Unix/Linux/Mac
            cmd = f'python {program_name}'
            subprocess.Popen(['gnome-terminal', '--', 'python', program_name])
        
        print(f"\n{program_name} iniciado en nueva terminal!")
        return True
    except Exception as e:
        print(f"Error al ejecutar {program_name}: {e}")
        return False

def main():
    # Lista de programas en orden de ejecución
    programs = [
        "InputGUI.py",          # Envía datos EEG a través de AURA_Power
        "realtime_processor.py", # Procesa datos y envía a FEATURE_STREAM
        "cnn_interference.py",   # Recibe FEATURE_STREAM y envía CNN_COMMANDS
        "CommandSender.py"      # Recibe comandos de CNN_COMMANDS
    ]
    
    print("Iniciando programas en terminales separadas...")
    
    # Ejecutar cada programa en una nueva terminal con un pequeño delay
    for program in programs:
        success = run_program_in_new_terminal(program)
        if not success:
            print(f"Error al iniciar {program}")
            sys.exit(1)
        # Esperar 2 segundos entre cada inicio para permitir que los streams LSL se inicialicen
        time.sleep(2)
    
    print("\nTodos los programas han sido iniciados en terminales separadas!")
    print("\nFlujo de datos LSL:")
    print("1. InputGUI.py -> AURA_Power stream")
    print("2. realtime_processor.py -> FEATURE_STREAM")
    print("3. cnn_interference.py -> FEATURE_STREAM")
    print("4. CommandSender.py <- CNN_COMMANDS")
    print("\nPuede cerrar cada programa individualmente en su terminal correspondiente.")

if __name__ == "__main__":
    main() 