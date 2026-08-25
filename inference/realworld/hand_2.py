import serial
import time


regdict = {
    "ID": 1,
    "baudrate": 7,
    "clearErr": 8,
    "angleSet": 36,
    "angleAct": 38,
    "speedSet": 35,
    "modeSet": 32,
    "forceSet": 34,
    "forcedirection": 21,
    "forceAct": 1582,
    "errCode": 1606,
    "statusCode": 1612,
    "temp": 1618,
    "actionSeq": 2320,
    "actionRun": 2322,
}

POSITION_TIMEOUT_S = 2.0
SERIAL_POLL_INTERVAL_S = 0.001
RESPONSE_HEADER = b"\xAA\x55"


def _read_response_frame(ser):
    deadline = time.monotonic() + POSITION_TIMEOUT_S
    received = bytearray()

    while time.monotonic() < deadline:
        received.extend(ser.read_all())
        header_index = received.find(RESPONSE_HEADER)
        if header_index >= 0:
            del received[:header_index]
            if len(received) >= 3:
                frame_size = received[2] + 5
                if len(received) >= frame_size:
                    frame = bytes(received[:frame_size])
                    if sum(frame[2:-1]) & 0xFF != frame[-1]:
                        raise ValueError(f"invalid gripper checksum: {frame.hex(' ')}")
                    return frame
        time.sleep(SERIAL_POLL_INTERVAL_S)

    raise TimeoutError(
        f"gripper response timed out after {POSITION_TIMEOUT_S:.1f}s: "
        f"{received.hex(' ')}"
    )

def openSerial(port, baudrate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    return ser


def closeSerial(ser):
    ser.close()


def writeRegister(ser, add, id, num, val):
    bytes = [0x55, 0xAA]
    bytes.append(num + 3)
    bytes.append(id)
    bytes.append(0x31)
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF)
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    ser.read_all()


def readRegister(ser, id, add, num, mute=False):
    bytes = [0x55, 0xAA]
    bytes.append(0x03)
    bytes.append(0x01)
    bytes.append(0x30)
    bytes.append(0x00)
    bytes.append(0x00)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(2)
    recv = ser.read_all()
    if len(recv) == 0:
        return []
    num = (recv[3] & 0xFF) - 3
    val = []
    for i in range(num):
        val.append(recv[7 + i])
    if not mute:
        print("register values:", *val)
    return val


def write(ser, str, id, val):
    if str == "angleSet" or str == "forceSet" or str == "speedSet":
        val_reg = []
        for i in range(1):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, regdict[str], id, 2, val_reg)
    else:
        print("str must be angleSet, forceSet, or speedSet")


def read(ser, id, str):
    if (
        str == "angleSet"
        or str == "forceSet"
        or str == "speedSet"
        or str == "angleAct"
        or str == "forceAct"
    ):
        val = readRegister(ser, id, regdict[str], 12, True)
        if len(val) < 12:
            print("no data received")
            return
        val_act = []
        for i in range(6):
            val_act.append((val[2 * i] & 0xFF) + (val[1 + 2 * i] << 8))
        print("register values:", *val_act)
    elif str == "errCode" or str == "statusCode" or str == "temp":
        val_act = readRegister(ser, id, regdict[str], 6, True)
        if len(val_act) < 6:
            print("no data received")
            return
        print("register values:", *val_act)
    else:
        print("unsupported register")


def pinjie(L, H):
    L = dec_to_hex(L)
    H = dec_to_hex(H)
    res = H + L[2:]
    res = hex_to_dec(res)
    return res


def read_hand(ser, mute=False):
    bytes = [0x55, 0xAA]
    bytes.append(0x03)
    bytes.append(0x01)
    bytes.append(0x30)
    bytes.append(0x00)
    bytes.append(0x00)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(2)
    recv = ser.read_all()

    position = pinjie(recv[7], recv[8])
    print("position", position)
    current = pinjie(recv[9], recv[10])
    print("current", current)
    force = pinjie(recv[11], recv[12])
    print("force", force)
    speed = pinjie(recv[13], recv[14])
    print("speed", speed)
    status = pinjie(recv[15], recv[16])
    print("status", status)


def read_reg(ser, mute=False):
    bytes = [0x55, 0xAA]
    bytes.append(0x04)
    bytes.append(0x01)
    bytes.append(0x32)
    bytes.append(0x00)
    bytes.append(0x26)
    bytes.append(0x02)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(2)
    ser.read_all()


def get_position(ser):
    request = bytes([0x55, 0xAA, 0x03, 0x01, 0x30, 0x00, 0x00, 0x34])
    ser.reset_input_buffer()
    ser.write(request)
    ser.flush()
    response = _read_response_frame(ser)
    return pinjie(response[7], response[8])


def hex_to_dec(hex_num):
    return int(hex_num, 16)


def dec_to_hex(dec_num):
    return hex(dec_num)
