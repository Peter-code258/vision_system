# src/sensors/ir_reader.py
import serial, threading, time

class IRSerialReader:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1):
        self.port = port; self.baudrate = baudrate; self.timeout = timeout
        self.ser = None; self.running=False; self.latest={}; self.lock=threading.Lock()

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except Exception as e:
            print("[IRSerialReader] open error:", e); self.ser=None; return
        self.running=True
        threading.Thread(target=self._loop, daemon=True).start()

    def _parse(self, line):
        try:
            line=line.strip()
            if ":" in line:
                k,v=line.split(":",1); v=v.strip()
                try: v = float(v)
                except: pass
                with self.lock: self.latest[k.upper()] = v
            else:
                with self.lock: self.latest['RAW'] = line
        except Exception:
            pass

    def _loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline().decode(errors='ignore')
                if raw:
                    self._parse(raw)
            except Exception:
                time.sleep(0.1)

    def stop(self):
        self.running=False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def get_latest(self):
        with self.lock:
            return dict(self.latest)