# CACI BITS-IRAD Beholder
# Author: Preston Walraven 6/22

import json
import socket
import threading


class TCP():
    """A class to establish a TCP connection using sockets.
    """
    def __init__(self, ip: str = '127.0.0.1', port: int = 8080):
        """Set the IP and Port and attempt to connect.

        Args:
            ip (str, optional): Target IP address. Defaults to '127.0.0.1'.
            port (int, optional): Target port. Defaults to 8080.
        """
        self.ip = ip
        self.port = port
        self.conn = None
        self.connect()

    def connect(self):
        """Attempt to connect in a server setting.
        """
        print('Establishing socket...')
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # sock.bind((self.ip, self.port))
        # sock.listen(1)
        # self.conn, addr = sock.accept()
        self.conn.connect((self.ip, self.port))
        # print(f"Connected by {addr}")
        listen_thread = threading.Thread(target=self.listen_thread)
        listen_thread.setDaemon(True)
        listen_thread.start()  # TODO need to listen?

    def listen_thread(self):
        """Prints any messages received from the socket.
        """
        while True:
            try:
                data = self.conn.recv(1024)
                if data:
                    print(f'RCVD: {data}')
            except ConnectionAbortedError:
                return False

    def send_json(self, json_obj):
        """Send an encoded json object to the connected socket.

        Args:
            json_obj (_type_): The object to send.
        """
        message = json.dumps(json_obj).encode()
        self.conn.send(message)
        print(f"SENT: {message!r}")

    def send_message(self, message):
        """Send an encoded message to the connected socket.

        Args:
            message (str): The message to send.
        """
        message = message.encode()
        self.conn.send(message)
        print(f"SENT: {message}")

    def cleanup(self):
        """Close the connected socket.
        """
        self.conn.close()
        print('Connection closed.')


if __name__ == '__main__':
    # example usage
    tcp = TCP()
    tcp.send_json({'x': 5, 'y': 6})
    input()  # wait for cleanup
    tcp.send_json({'x': 7, 'y': 8})
    input()  # wait for cleanup
    tcp.send_json({'v3': [1, 2, 3]})
    tcp.cleanup()
