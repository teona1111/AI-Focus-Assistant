import socket
import threading
import serial
import serial.tools.list_ports
import time
import os

from parser import decode_file

HOST = "0.0.0.0"
PORT = 5000

VID = "0483"
PID = "5740"

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

ser = None
stm_connected = False
stm_port = None

clients = []

def get_file_list():

    ser.reset_input_buffer()

    ser.write(b"LIST\n")

    time.sleep(1)

    data = ser.read_all().decode(
        errors="ignore"
    )

    files = []

    for line in data.splitlines():

        line = line.strip()

        if line.startswith("LOG") and ".BIN" in line:

            filename = line.split()[0]

            files.append(filename)

    return sorted(files)

def connect_stm32():
    global ser
    global stm_connected
    global stm_port

    while True:
        if not stm_connected:
            ports = serial.tools.list_ports.comports()

            for port in ports:
                hwid = port.hwid

                if VID in hwid and PID in hwid:
                    try:
                        ser = serial.Serial(port.device, 115200, timeout=1)

                        stm_connected = True
                        stm_port = port.device

                        print(f"STM32 detected at {port.device}")

                        broadcast(f"STM32 detected at {port.device}")

                    except Exception as e:
                        print(f"Connection failed: {e}")

        else:
            ports = serial.tools.list_ports.comports()
            found = False

            for port in ports:
                if port.device == stm_port:
                    found = True
                    break

            if not found:
                disconnect_stm32()
            

        time.sleep(2)

def disconnect_stm32():
    global ser
    global stm_connected
    global stm_port

    try:
        if ser:
            ser.close()
    except:
        pass

    stm_connected = False
    stm_port = None
    ser = None

    print("STM32 has disconnected")

    broadcast("STM32 has disconnected")

def broadcast(msg):
    global clients

    dead = []

    for c in clients:
        try:
            c.sendall((msg + "\n").encode())
        except:
            dead.append(c)

    for c in dead:
        if c in clients:
            clients.remove(c)

def handle_client(conn, addr):
    global clients

    print(f"Client connected: {addr}")

    clients.append(conn)

    try:
        if stm_connected:
            conn.sendall(b"Connected to SPO STM32 service\n")
        else:
            conn.sendall(
                b"Connected to SPO STM32 service - No STM32 detected\n"
            )

        while True:
            data = conn.recv(1024)

            if not data:
                break

            cmd = data.decode().strip()

            response = process_command(cmd)

            conn.sendall((response + "\n").encode())

    except Exception as e:
        print(e)

    finally:
        if conn in clients:
            clients.remove(conn)

        conn.close()

def process_command(cmd):
    global ser

    if cmd == "STATUS":
        if stm_connected:
            return "STM32 is connected"
        return "FAIL: STM32 is not connected"

    if not stm_connected:
        return "FAIL: STM32 is not connected"

    try:
        if cmd == "GET_LAST":

            files = get_file_list()

            if not files:
                return "FAIL: No files found"
            
            last_file = files[-1]

            ser.write(
                f"GET {last_file}\n".encode()
            )
            
            receive_file(last_file)

            decode_file(
                os.path.join(DOWNLOAD_DIR, last_file)
            )

            return f"Last file ({last_file}) from STM32 has been processed"

        elif cmd == "GET_ALL":

            files = get_file_list()

            if not files:
                return "FAIL: No files found"
            
            for filename in files:
                ser.write(
                    f"GET {filename}\n".encode()
                )

                receive_file(filename)

                decode_file(
                    os.path.join(DOWNLOAD_DIR, filename)
                )

            return "All files from STM32 are processed"

        elif cmd.startswith("GET_FILE|"):
            
            filename = cmd.split("|", 1)[1]

            files = get_file_list()

            if filename not in files:
                return f"FAIL: File {filename} not found on STM32"
        
            ser.write(f"GET {filename}\n".encode())

            receive_file(filename)

            decode_file(
                os.path.join(DOWNLOAD_DIR, filename)
            )

            return f"File {filename} from STM32 has been processed"

        elif cmd == "DELETE":

            ser.write(b"DELETE\n")

            return "All files on STM32 are deleted"

        else:
            return "FAIL: Unknown command"

    except Exception as e:
        print(e)

        disconnect_stm32()

        return "FAIL: STM32 communication error"

def receive_file(filename):
    global ser

    path = os.path.join(DOWNLOAD_DIR, filename)

    with open(path, "wb") as f:

        timeout = time.time() + 5

        while True:

            if time.time() > timeout:
                break

            data = ser.read(1024)

            if data:
                f.write(data)
                timeout = time.time() + 2
            else:
                break

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, PORT))

    server.listen()
    server.settimeout(1)

    print(f"Service listening on port {PORT}")

    while True:
        try:
            conn, addr = server.accept()

            thread = threading.Thread(
                target=handle_client,
                args=(conn, addr)
            )

            thread.start()

        except socket.timeout:
            continue

def main():
    threading.Thread(
        target=connect_stm32,
        daemon=True
    ).start()

    start_server()

if __name__ == "__main__":
    main()