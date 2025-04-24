#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CommandSender.py
Escucha CNN_COMMANDS (LSL) y envía al dron Tello el comando asociado a cada clase.
"""

import socket
import time
import logging
from pylsl import resolve_byprop, StreamInlet

# — Configuración de logging —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("command_sender_log.txt"),
        logging.StreamHandler()
    ]
)

# — Configuración del dron Tello —
TELLO_IP      = "192.168.10.1"
TELLO_PORT    = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 9000))

def send_command(cmd: str, wait: float = 0.5):
    """Envía un comando al Tello y espera un instante."""
    logging.info(f"→ {cmd}")
    sock.sendto(cmd.encode("utf-8"), TELLO_ADDRESS)
    time.sleep(wait)

# — Mapeo de clase EEG → comando Tello —
COMMAND_MAP = {
    "LeftArmThinking":  "forward 50",
    "RightArmThinking": "back 50",
    "LeftFistThinking": "flip l",   # al menos un flip
    "RightFistThinking":"flip r"
}

def main():
    # 1) Activar modo SDK y despegar
    send_command("command", wait=1)
    send_command("takeoff", wait=5)

    # 2) Conectar al stream LSL CNN_COMMANDS
    streams = resolve_byprop("name", "CNN_COMMANDS", timeout=5)
    if not streams:
        logging.error("No encontré ningún stream llamado 'CNN_COMMANDS'")
        return
    inlet = StreamInlet(streams[0])
    logging.info("Escuchando CNN_COMMANDS…")

    try:
        while True:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if not sample:
                continue

            label = sample[0]  # p. ej. "RightArmThinking"
            cmd = COMMAND_MAP.get(label)
            if cmd:
                send_command(cmd, wait=2)
            else:
                logging.warning(f"Clase desconocida recibida: {label}")

    except KeyboardInterrupt:
        # Al interrumpir, aterrizo y cierro socket
        logging.info("Interrupción recibida, aterrizando…")
        send_command("land", wait=5)
    finally:
        sock.close()
        logging.info("Socket cerrado, fin de CommandSender.py")

if __name__ == "__main__":
    main()
