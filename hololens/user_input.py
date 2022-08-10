# import keyboard
from dataclasses import dataclass
import time
import json
import random

from server import TCP


@dataclass
class Location:
    # lat: float
    # lon: float
    # alt: float
    x: float
    y: float
    z: float

    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_list(self):
        return [round(self.x, 3), round(self.y, 3), round(self.z, 3)]

    def to_tuple(self):
        return (round(self.x, 3), round(self.y, 3), round(self.z, 3))

    # TODO add overflow for lat and lon changes

    # def to_list(self):
    #     return [self.lat, self.lon, self.alt]

    # def to_ecef(self):
    #     return geodetic2ecef(self.lat, self.lon, self.alt)


@dataclass
class Target:
    posn: Location
    cls: str
    id: int


class UserInput:
    def __init__(self, tcp: TCP):
        self.tcp = tcp
        self.targets = []
        # threading.Thread(target=self.read_user_input).start()

    def add_target(self, target):
        self.targets.append(target)

    def play_scene(self):
        # while True:
        #     for target in self.targets:
        #         target.posn.set(random.randint(-50, 50),
        #                         random.randint(-50, 50),
        #                         random.randint(-50, 50))

        #     self.send_update()
        #     time.sleep(0.01)

        while True:
            x1, y1, z1 = 3, 0, 0
            x2, y2, z2 = -3, 0, 20
            while z1 < 20:
                self.targets[0].posn.set(x1, y1, z1)
                self.targets[1].posn.set(x2, y2, z2)
                self.send_update()
                z1 += 0.05
                z2 -= 0.05
                time.sleep(0.01)

    def send_update(self):
        obj = {
            "tracks": [
                {
                    "point": target.posn.to_tuple(),
                    "track_id": f'{target.id}',
                    "label_name": f'{target.cls}'
                } for target in self.targets
            ]
        }

        # to simulate broken json objects
        temp_str = json.dumps(obj) + '\n'
        messages = []
        idx = 0
        while idx < len(temp_str):
            offset = random.randint(0, len(temp_str) - idx)
            messages.append(temp_str[idx:idx + offset])
            idx += offset

        for message in messages:
            self.tcp.send_message(message)

    # def read_user_input(self):
    #     """TODO: Should not be used in final demo, for testing purposes
    #     only."""
    #     while True:
    #         send_update = False
    #         if keyboard.is_pressed('w'):  # north
    #             send_update = True
    #             self.targets[0].posn.y += 0.1
    #             self.targets[1].posn.y -= 0.1
    #         if keyboard.is_pressed('a'):  # west
    #             send_update = True
    #             self.targets[0].posn.x -= 0.1
    #             self.targets[1].posn.x += 0.1
    #         if keyboard.is_pressed('s'):  # south
    #             send_update = True
    #             self.targets[0].posn.y -= 0.1
    #             self.targets[1].posn.y += 0.1
    #         if keyboard.is_pressed('d'):  # east
    #             send_update = True
    #             self.targets[0].posn.x += 0.1
    #             self.targets[1].posn.x -= 0.1
    #         if keyboard.is_pressed('e'):  # up
    #             send_update = True
    #             self.targets[0].posn.z += 0.1
    #             self.targets[1].posn.z -= 0.1
    #         if keyboard.is_pressed('q'):  # down
    #             send_update = True
    #             self.targets[0].posn.z -= 0.1
    #             self.targets[1].posn.z += 0.1
    #         if keyboard.is_pressed('l'):  # quit
    #             send_update = True
    #             self.tcp.cleanup()
    #             exit(1)
    #         if keyboard.is_pressed('p'):
    #             self.targets.pop()
    #             self.add_target(Target(posn=Location(0,0,0),cls='car',id=1))
    #             time.sleep(1)
    #             send_update = True

    #         if send_update:
    #             self.send_update()


def main():
    # tcp = TCP(ip='192.168.137.193', port=8080)
    # print([ip for ip in socket.gethostbyname_ex(socket.gethostname())])
    # print(socket.gethostbyname('DESKTOP-R2LATVH'))
    # hololens_ip = socket.gethostbyname('HOLOLENS-QRLEEI')
    # print(hololens_ip)
    # tcp = TCP(ip='169.254.204.136', port=8080)
    # tcp = TCP(ip='172.28.114.86', port=8080)

    # tcp = TCP(ip='172.20.10.4', port=8080)
    tcp = TCP()
    ui = UserInput(tcp)

    t1 = Target(posn=Location(0, 0, 0), cls='car', id=0)
    t2 = Target(posn=Location(0, 0, 0), cls='car', id=1)
    ui.add_target(t1)
    ui.add_target(t2)

    # for i in range(100):
    #     ui.add_target(Target(posn=Location(0, 0, 0), cls='car', id=i))

    ui.play_scene()


if __name__ == '__main__':
    main()
