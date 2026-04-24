import serial
import csv
import time
import math
from datetime import datetime

port = "/dev/cu.usbserial-10"
baud = 9600
duration = 60  # 1 minute

outfile = "imu_data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

G = 9.80665
DEG2RAD = math.pi / 180

def s16(lo, hi):
    v = lo | (hi << 8)
    if v >= 32768:
        v -= 65536
    return v

ax = ay = az = 0.0
rows = 0
accel_packets = 0
gyro_packets = 0
angle_packets = 0

start = time.time()

ser = serial.Serial(port, baud, timeout=0.05)
ser.reset_input_buffer()

try:
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])

        buf = bytearray()

        while time.time() - start < duration:
            data = ser.read(1000)
            if data:
                buf.extend(data)

            while len(buf) >= 11:
                idx = buf.find(b"\x55")
                if idx == -1:
                    buf.clear()
                    break

                del buf[:idx]

                if len(buf) < 11:
                    break

                pkt = buf[:11]
                typ = pkt[1]

                if typ == 0x51:
                    accel_packets += 1
                    ax = s16(pkt[2], pkt[3]) / 32768 * 16 * G
                    ay = s16(pkt[4], pkt[5]) / 32768 * 16 * G
                    az = s16(pkt[6], pkt[7]) / 32768 * 16 * G
                    del buf[:11]

                elif typ == 0x52:
                    gyro_packets += 1
                    gx = s16(pkt[2], pkt[3]) / 32768 * 2000 * DEG2RAD
                    gy = s16(pkt[4], pkt[5]) / 32768 * 2000 * DEG2RAD
                    gz = s16(pkt[6], pkt[7]) / 32768 * 2000 * DEG2RAD

                    t = time.time() - start
                    w.writerow([t, ax, ay, az, gx, gy, gz])
                    rows += 1
                    del buf[:11]

                elif typ == 0x53:
                    angle_packets += 1
                    del buf[:11]

                else:
                    del buf[0]

        f.flush()

finally:
    ser.close()

elapsed = time.time() - start

print("Done.")
print(f"File: {outfile}")
print(f"Rows written: {rows}")
print(f"Accel packets: {accel_packets}")
print(f"Gyro packets: {gyro_packets}")
print(f"Angle packets: {angle_packets}")
print(f"Elapsed: {elapsed:.2f} sec")
print(f"Approx row rate: {rows / elapsed:.2f} Hz")