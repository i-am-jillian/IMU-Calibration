import serial
import time

port = "/dev/cu.usbserial-10"  # change if check_ports.py shows a new one
bauds = [4800, 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

for baud in bauds:
    print("\nTesting baud:", baud)

    try:
        ser = serial.Serial(port, baud, timeout=1)
        ser.reset_input_buffer()
        time.sleep(1)

        data = ser.read(1000)
        ser.close()

        print("Length:", len(data))
        print("0x55 count:", data.count(b"\x55"))
        print("Contains accel UQ:", b"UQ" in data)
        print("Contains gyro UR:", b"UR" in data)
        print("Contains angle US:", b"US" in data)
        print("First bytes:", data[:60])

    except Exception as e:
        print("Error:", e)