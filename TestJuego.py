import socket
import time

TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 9000))

def send_command(command, wait=0):
    print(f"→ {command}")
    sock.sendto(command.encode(), TELLO_ADDRESS)
    time.sleep(wait)

try:
    send_command('command', wait=1)       # Activar modo SDK
    send_command('takeoff', wait=5)       # Despegue
    time.sleep(2)
    send_command('up 50', wait=3)         # Subir 50 cm
    time.sleep(2)
    send_command('forward 100', wait=3)   # Avanzar 100 cm
    time.sleep(2)
    send_command('cw 90', wait=3)         # Girar 90° en sentido horario
    time.sleep(2)
    send_command('back 100', wait=3)        # Retroceder 100 cm
    time.sleep(2)
    send_command('ccw 90', wait=3)        # Girar 90° en sentido antihorario
    time.sleep(2)
    send_command('down 50', wait=3)       # Bajar 50 cm
    time.sleep(2)
    send_command('flip f', wait=3)       # Hacer un flip hacia adelante
    time.sleep(2)
    send_command('flip b', wait=3)       # Hacer un flip hacia atrás
    
    #send_command('flip l', wait=3)       # Hacer un flip hacia la izquierda
    #send_command('flip r', wait=3)       # Hacer un flip hacia la derecha
    #send_command('flip fw', wait=3)      # Hacer un flip hacia adelante y hacia la derecha
    #send_command('flip bw', wait=3)      # Hacer un flip hacia atrás y hacia la izquierda
    time.sleep(2)
    send_command('land', wait=5)          # Aterrizaje

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    sock.close()
