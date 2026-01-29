
class RobotARM165:
    def __init__(self, ip="192.168.1.100", port=8080):
        self.ip = ip
        self.port = port
        self.connected = False
        print(f"Инициализация робота: {ip}:{port}")
    
    def connect(self):
        #code connact
        print(f"ПОДКЛЮЧЕНИЕ К РОБОТУ: {self.ip}:{self.port}")
        self.connected = True  # zaglush
        return True
    
    def disconnect(self):
        #disconext
        self.connected = False
        return True
    
    def move_to_point(self, x, y, z):
        #dviz v tochke
        if not self.connected:
            return False
        
        # dvizh
        
        cmd = f"MOVE_L {x} {y} {z}\n"
        self.socket.send(cmd.encode())
        response = self.socket.recv(1024)
        return response == b"OK"
        #####################################################################################
        print(f"РЕАЛЬНАЯ КОМАНДА: Движение к X={x}, Y={y}, Z={z}")
        return True
    
    def gripper_on(self):
        #vkl gripper
        if not self.connected:
            return False
        
        # zahvat vkl
        print("РЕАЛЬНАЯ КОМАНДА: Захват ВКЛЮЧЕН")
        return True
    
    def gripper_off(self):
        # zahvat vikl
        if not self.connected:
            return False
        
        # zahvat
        print("РЕАЛЬНАЯ КОМАНДА: Захват ВЫКЛЮЧЕН")
        return True
    
    def set_joint_angles(self, angles):
        # ugli zahvata
        if not self.connected:
            return False
        
        # susta
        print(f"РЕАЛЬНАЯ КОМАНДА: Углы суставов {angles}")
        return True
    
    def manual_cart_mode(self):
        #sam uprav v dekalt tochkah
        if not self.connected:
            return False
        
        # MANUAL CART
        print("РЕАЛЬНАЯ КОМАНДА: Режим Manual Cart")
        return True
    
    def manual_joint_mode(self):
        # sam rukami
        if not self.connected:
            return False
        
        # MANUAL JOINT
        print("РЕАЛЬНАЯ КОМАНДА: Режим Manual Joint")
        return True
    
    def get_position(self):
        #gde ti seichas
        if not self.connected:
            return None

        self.socket.send(b"GET_POSITION\n")
        response = self.socket.recv(1024)
        return self.parse_position(response)
        
   