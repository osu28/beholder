# See TCP server in server.py instead
# Only for testing purposes

import asyncio

ip = '127.0.0.1'
# ip = '172.18.32.1'
port = 8888


async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(ip, port)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()

    data = await reader.read(100)
    print(f'Received: {data.decode()!r}')

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

asyncio.run(tcp_echo_client('Hello World!'))
