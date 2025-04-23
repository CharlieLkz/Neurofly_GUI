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
    send_command('land', wait=5)          # Aterrizaje

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    sock.close()
