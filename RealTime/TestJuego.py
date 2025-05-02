import socket
import time
import sys
import threading

TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
LOCAL_PORT = 9000

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', LOCAL_PORT))

# Set timeout for receiving responses
sock.settimeout(5.0)

# Timeout para aterrizaje automático (15 segundos)
INACTIVITY_TIMEOUT = 15.0

# Lista completa de comandos del Tello
TELLO_COMMANDS = {
    # Comandos básicos
    'command': 'Entrar en modo SDK',
    'takeoff': 'Despegar',
    'land': 'Aterrizar',
    'streamon': 'Activar streaming de video',
    'streamoff': 'Desactivar streaming de video',
    'emergency': 'Detener motores inmediatamente',
    
    # Movimiento
    'up 20': 'Ascender 20 cm',
    'down 20': 'Descender 20 cm',
    'left 20': 'Mover a la izquierda 20 cm',
    'right 20': 'Mover a la derecha 20 cm',
    'forward 20': 'Avanzar 20 cm',
    'back 20': 'Retroceder 20 cm',
    
    # Rotación
    'cw 90': 'Girar 90 grados en sentido horario',
    'ccw 90': 'Girar 90 grados en sentido antihorario',
    
    # Flips
    'flip l': 'Flip a la izquierda',
    'flip r': 'Flip a la derecha',
    'flip f': 'Flip hacia adelante',
    'flip b': 'Flip hacia atrás',
    
    # Velocidad
    'speed 50': 'Establecer velocidad a 50 cm/s',
    
    # Batería
    'battery?': 'Consultar nivel de batería',
    
    # Tiempo
    'time?': 'Consultar tiempo de vuelo',
    
    # Altura
    'height?': 'Consultar altura',
    
    # Temperatura
    'temp?': 'Consultar temperatura',
    
    # Actitud
    'attitude?': 'Consultar actitud (pitch, roll, yaw)',
    
    # Barómetro
    'baro?': 'Consultar presión barométrica',
    
    # Aceleración
    'acceleration?': 'Consultar aceleración',
    
    # TOF (Time of Flight)
    'tof?': 'Consultar distancia al suelo'
}

def send_command(command, wait=0):
    """Send a command to the Tello drone and wait for a response"""
    print(f"> {command}")
    try:
        # Enviar comando
        sock.sendto(command.encode(), TELLO_ADDRESS)
        
        # Esperar respuesta
        response, ip = sock.recvfrom(1024)
        response = response.decode('utf-8').strip()
        print(f"< {response}")
        
        # Verificar respuesta
        if response.upper() != 'OK' and 'error' not in response.lower():
            print(f"! Unexpected response: {response}")
        
        time.sleep(wait)
        return response
    except socket.timeout:
        print("! No response received from drone")
        return None
    except Exception as e:
        print(f"! Error sending command: {e}")
        return None

def check_connection():
    """Check if the drone is connected and responsive"""
    print("Checking connection to Tello drone...")
    
    # Intentar conexión hasta 3 veces
    for attempt in range(3):
        try:
            # Enviar comando de prueba
            response = send_command('command', wait=1)
            if response and response.upper() == 'OK':
                print("+ Connected to Tello drone")
                return True
            else:
                print(f"- Connection attempt {attempt + 1} failed")
                if attempt < 2:  # Si no es el último intento
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
        except Exception as e:
            print(f"- Connection error: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    
    print("- Failed to connect to Tello drone after multiple attempts")
    return False

def monitor_inactivity():
    """Monitor inactivity and land if timeout is reached"""
    last_activity = time.time()
    while True:
        current_time = time.time()
        if current_time - last_activity > INACTIVITY_TIMEOUT:
            print("\n¡Timeout de inactividad detectado!")
            print("Iniciando aterrizaje automático...")
            try:
                send_command('land', wait=4)
            except:
                pass
            print("Programa terminado por inactividad")
            sys.exit(0)
        time.sleep(1)

def test_command(command, description):
    """Test a single command and return the result"""
    print(f"\n=== Probando comando: {command} ===")
    print(f"Descripción: {description}")
    response = send_command(command, wait=5)
    if response and response.upper() == 'OK':
        print(f"+ Comando {command} ejecutado exitosamente")
        return True
    else:
        print(f"- Error al ejecutar comando {command}")
        return False

def main():
    try:
        # Iniciar el monitoreo de inactividad
        inactivity_monitor = threading.Thread(target=monitor_inactivity, daemon=True)
        inactivity_monitor.start()
        
        # Verificar conexión
        if not check_connection():
            print("Exiting due to connection failure")
            sys.exit(1)
        
        # Test de comandos
        print("\nIniciando prueba de comandos del Tello...")
        
        # Primero, verificar estado del dron
        print("\n=== Verificando estado del dron ===")
        test_command('battery?', 'Consultar batería')
        test_command('time?', 'Consultar tiempo de vuelo')
        test_command('height?', 'Consultar altura')
        test_command('temp?', 'Consultar temperatura')
        
        # Comandos de movimiento básico
        print("\n=== Probando comandos de movimiento básico ===")
        test_command('takeoff', 'Despegar')
        time.sleep(5)  # Esperar a que el dron se estabilice
        
        # Movimientos verticales
        test_command('up 20', 'Ascender 20 cm')
        time.sleep(3)
        test_command('down 20', 'Descender 20 cm')
        time.sleep(3)
        
        # Movimientos horizontales
        test_command('left 20', 'Mover a la izquierda 20 cm')
        time.sleep(3)
        test_command('right 20', 'Mover a la derecha 20 cm')
        time.sleep(3)
        test_command('forward 20', 'Avanzar 20 cm')
        time.sleep(3)
        test_command('back 20', 'Retroceder 20 cm')
        time.sleep(3)
        
        # Rotaciones
        test_command('cw 90', 'Girar 90 grados en sentido horario')
        time.sleep(3)
        test_command('ccw 90', 'Girar 90 grados en sentido antihorario')
        time.sleep(3)
        
        # Flips
        test_command('flip l', 'Flip a la izquierda')
        time.sleep(5)
        test_command('flip r', 'Flip a la derecha')
        time.sleep(5)
        test_command('flip f', 'Flip hacia adelante')
        time.sleep(5)
        test_command('flip b', 'Flip hacia atrás')
        time.sleep(5)
        
        # Velocidad
        test_command('speed 50', 'Establecer velocidad a 50 cm/s')
        time.sleep(2)
        
        # Verificar estado después de los movimientos
        print("\n=== Verificando estado después de los movimientos ===")
        test_command('battery?', 'Consultar batería')
        test_command('time?', 'Consultar tiempo de vuelo')
        test_command('height?', 'Consultar altura')
        test_command('temp?', 'Consultar temperatura')
        
        # Aterrizar
        print("\n=== Finalizando prueba ===")
        test_command('land', 'Aterrizar')
        
        print("\nPrueba de comandos completada")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        try:
            send_command('land', wait=10)
        except:
            pass
    except Exception as e:
        print(f"X Error: {e}")
        try:
            send_command('land', wait=10)
        except:
            pass
    finally:
        print("Closing connection...")
        sock.close()

if __name__ == "__main__":
    main()

