from pyModbusTCP.client import ModbusClient

if __name__ == '__main__':
    c = ModbusClient(host="192.168.0.2", port=502, auto_open=True)
    c.open()
    regs = c.read_holding_registers(0, 2)
    if regs:
        print(regs)
    else:
        print("read error")