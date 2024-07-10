import face_recognition
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
from scipy.spatial import distance as dist
import serial
from flask import Flask, Response
import threading
from queue import Queue
from threading import Lock
import socket
import pygame
def send_signal_and_wait_for_response():
    # 树莓派服务器的IP地址和端口号
    server_ip = '192.168.2.102'
    server_port = 12345

    # 创建一个socket对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 连接到树莓派服务器
        s.connect((server_ip, server_port))
        
        # 向服务器发送信号，这里使用'start_recognition'作为示例
        s.sendall(b'start_recognition')
        print("Signal sent. Waiting for response...")
        
        # 等待并接收服务器的响应
        response = s.recv(1024)
        result = response.decode()
        # 打印并返回识别结果
        print('Received response:', result)
        
        return result

# Get a reference to webcam #0 (the default one)
app = Flask(__name__)
model_path='model/model3.ckpt'
serial_port_lock = Lock()
last_sent_time = time.time()
send_interval = 5  # 至少10秒才能再次发送数据

def read_test_data(test_dir):
    datas = []
    fpaths = []
    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        image = image.resize((50, 50), Image.BILINEAR)  # Resize images to 50x50
        data = np.array(image) / 255.0  # 归一化(Normalization)
        datas.append(data)
    datas = np.array(datas)
    print("Datas shape:", datas.shape)
    return fpaths, datas

# 计算人眼纵横比
def get_ear(eye):
    # 计算眼睛轮廓垂直方向上下关键点的距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平方向上的关键点的距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算眼睛的纵横比
    ear = (A + B) / (2.0 * C)

    # 返回眼睛的纵横比
    return ear

def calculate_checksum(bytes_list):
    """计算校验和"""
    return sum(bytes_list) & 0xFF

def open_serial_connection(port='/dev/ttyUSBA', baud_rate=9600, timeout=1):
    """
    尝试打开串行端口连接。
    """
    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(2)  # 给设备一些时间来响应
        print(f"Serial port {port} opened successfully by this program.")
        return ser
    except serial.SerialException as e:
        print(f"Serial port {port} might be in use by another program or could not be found. Error: {e}")
        return None

def send_data(ser, data):
    """
    向串行端口发送数据。
    """
    if ser is not None:
        try:
            ser.write(data)
            print("Data sent successfully.")
            time.sleep(2)  # 给设备一些时间来处理发送的数据
        except serial.SerialException as e:
            print(f"Failed to send data: {e}")
            
def close_serial_connection(ser):
    """
    关闭串行端口连接。
    """
    if ser is not None:
        ser.close()
        print("Serial connection closed.")

def read_response(ser):
    """
    从串行端口读取并打印所有可用的数据。
    """
    if ser is not None:
        try:
            # 检查缓冲区中等待的字节数
            bytes_to_read = ser.in_waiting
            
            # 如果缓冲区中有数据，读取所有这些数据
            if bytes_to_read:
                response = ser.read(bytes_to_read)
                #print(f"Received data: {response}")
            else:
                print("No data available to read.")
        except serial.SerialException as e:
            print(f"Failed to read response: {e}")

def safe_send_data_packet(ser, data):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            send_data_packet(ser, data)
            break  # 成功发送，跳出循环
        except serial.SerialException:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(2)  # 稍等一会再重试
            try:
                ser.close()  # 尝试关闭当前串行连接
            except serial.SerialException:
                pass
            try:
                ser.open()  # 尝试重新打开串行连接
            except serial.SerialException as e:
                print(f"Failed to reopen serial port: {e}")
    else:
        print("Failed to send data packet after several attempts.")

def send_data_packet(ser, data):
    with serial_port_lock:
        try:
            reset_serial_connection(ser)
            function_code = 0x01
            data_length = len(data)
            checksum = calculate_checksum(data)
            packet = [function_code, data_length] + data + [checksum]
            data_to_send = b"\x01\x01\x01\x01"  # 准备要发送的数据
            send_data(ser, data_to_send)  # 发送数据
            #ser.write(bytearray(packet))
            print("Data packet sent:", packet)
            receive_response(ser)
        except serial.SerialException as e:
            print(f"Serial write failed: {e}")

            # 重新初始化串行连接或其他恢复逻辑

def reset_serial_connection(ser):
    """尝试重置串行连接"""
    try:
        if ser.is_open:
            ser.close()
        ser.open()
        time.sleep(1)  # 给串行设备一些时间来重置
        print("Serial connection reset successfully.")
    except serial.SerialException as e:
        print(f"Failed to reset serial connection: {e}")

def receive_response(ser):
    print("Waiting for response...")
    response = ser.read(2)  # 应答码 + 校验和
    print(f"Raw response: {response}")
    
    if len(response) == 2:
        resp_code, checksum = response
        print(f"Response code: {resp_code}, Checksum received: {checksum}")

        # 计算校验和并与接收到的校验和比较
        if calculate_checksum([resp_code]) == checksum:
            print("Checksum valid.")

            if resp_code == 0xFF:
                print("Operation success.")
            else:
                print("Operation failed.")
        else:
            print("Checksum error.")
    else:
        print("No response or incomplete response.")



def capture_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # 只保留最新的帧
        if frame_queue.full():
            frame_queue.get()  # 移除旧的帧
        frame_queue.put(frame)

# 全局变量
joystick_initialized = False
joystick = None  # 这将用来存储Joystick对象

def joystick_manager():
    global joystick_initialized
    global joystick  # 引用全局变量
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    while True:
        pygame.event.pump()  # 更新joystick的状态
        # 检查是否有joystick已连接
        if pygame.joystick.get_count() > 0:
            if not joystick_initialized:
                try:
                    joystick = pygame.joystick.Joystick(0)  # 尝试初始化第一个joystick
                    joystick.init()
                    print(f"Initialized Joystick : {joystick.get_name()}")
                    joystick_initialized = True
                except pygame.error as e:
                    print(f"Joystick initialization error: {e}")
                    joystick = None  # 重置joystick对象
        else:
            if joystick_initialized:
                print("Joystick disconnected.")
            joystick_initialized = False
            joystick = None
        time.sleep(5)  # 每5秒检查一次

def handle_joystick_input(lock, ser):
    global joystick  # 引用全局变量
    while True:
        if joystick_initialized and joystick is not None:
            pygame.event.pump()  # Update internal state of the event handler
            try:
                button = joystick.get_button(0)  # Example: read the state of the first button
                if button:
                    with lock:
                        print("Button pressed, sending data...")
                        data_to_send = b"\x01\x01\x01\x01"
                        send_data(ser, data_to_send)
                        read_response(ser)
            except pygame.error as e:
                print(f"Error reading joystick button: {e}")
        time.sleep(0.1)  # Adjust based on how often you want to check the joystick input
def image_processing():
    data_packet_sent = False
    process_this_frame = True
    closed_count=0
    #global frame_queue
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            if process_this_frame:
                start_time = time.time()
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_small_frame = cv2.cuda.resize(gpu_frame, (0, 0), fx=0.25, fy=0.25)
                

                gpu_rgb_small_frame = cv2.cuda.cvtColor(gpu_small_frame, cv2.COLOR_BGR2RGB)
                rgb_small_frame  = gpu_rgb_small_frame.download()


                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                if face_locations:
                    print("检测到人脸。")
                    # 进一步的处理，比如分析 face_encodings
                    current_time = time.time()
                    result = send_signal_and_wait_for_response()
                    if result == 'true':
                        print("Face recognition succeeded.")
                        if not data_packet_sent or (current_time - last_sent_time) > send_interval:
                            data_to_send = b"\x01\x01\x01\x01"  # 准备要发送的数据
                            send_data(ser, data_to_send)  # 发送数据
                            read_response(ser)  # 读取响应
                            #close_serial_connection(ser)  # 关闭串行端口
                            #safe_send_data_packet(ser, [0x01])  # 发送数据包
                            data_packet_sent = True  # 标记为已发送
                            last_sent_time = current_time  # 更新发送时间
                    elif result == 'false':
                        print("Face recognized but not a known person.")
                    elif result == 'empty':
                        print("No face detected.")
                    else:
                        print("Unknown response:", result)
                else:
                    print("未检测到人脸。")

                


            
            
def generate():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            success, jpeg = cv2.imencode('.jpg', frame)

            frame = jpeg.tobytes()
    
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return "<html><body><h1>Hello, this is a test message!</h1></body></html>"


video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
# 检查视频捕获设备是否成功打开
if video_capture.isOpened():
    print("视频设备成功打开.")
else:
    print("无法打开视频设备. 请检查设备连接和索引号.")



video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

desired_width = 640
desired_height = 480
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# 检查设置后的分辨率
actual_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"摄像头分辨率已设置为: {actual_width}x{actual_height}")
fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"当前FOURCC编码: {fourcc_str}")

# 创建一个队列来保存待推流的帧
frame_queue = Queue(maxsize=1)








if __name__ == '__main__':

    serial_port_lock = Lock()
    ser = open_serial_connection()  # 尝试打开串行端口
    
    t1 = threading.Thread(target=capture_frames)
    t1.daemon = True
    t1.start()
    
    # 创建并启动图像处理的线程
    t2 = threading.Thread(target=image_processing)
    t2.daemon = True
    t2.start()

    # 启动操纵杆管理线程
    joystick_thread = threading.Thread(target=joystick_manager)
    joystick_thread.daemon = True
    joystick_thread.start()

    # 启动处理操纵杆输入的线程
    joystick_input_thread = threading.Thread(target=handle_joystick_input, args=(serial_port_lock, ser))
    joystick_input_thread.daemon = True
    joystick_input_thread.start()

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, threaded=True)

    # 如果有必要，等待线程结束
    t1.join()
    t2.join()
    joystick_thread.join()
    joystick_input_thread.join()

    # 清理资源
    pygame.quit()
    if ser:
        ser.close()
    video_capture.release()


