"""
serial_bridge.py
----------------
Hardware UART Bridge for Xilinx VC707 FPGA (RISC-V SoC).
Handles COM port auto-discovery, live stream parsing, and offline hardware simulation.
"""

import time
import re
import numpy as np
from typing import Dict, List, Optional, Callable
import streamlit as st  # add this import at top

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def list_available_com_ports() -> List[tuple]:
    """Scans and returns all connected serial / COM ports with descriptions."""
    results = [("SIMULATION_MODE", "🟡 SIMULATION_MODE (Offline Presentation)")]
    if HAS_SERIAL:
        for p in serial.tools.list_ports.comports():
            # e.g., ("COM5", "COM5 - Standard Serial over Bluetooth link")
            desc = f"{p.device} - {p.description}"
            results.append((p.device, desc))
    return results



class VC707TelemetryParser:
    """Parses structured ASCII tokens from the VC707 UART stream."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.model = "VGG-11 (8-bit)"
        self.mode = "8-bit Uniform Baseline"
        self.current_layer = ""
        self.progress_pct = 0.0
        self.cycles = 0
        self.instructions = 0
        self.cpi = 0.0
        self.latency_ms = 0.0
        self.top5_scores: Dict[str, float] = {}
        self.pred_class_id = -1
        self.pred_class_name = ""
        self.status = "IDLE"
        self.raw_logs: List[str] = []

    def parse_line(self, line: str) -> Optional[str]:
        """Parses a single line of UART telemetry."""
        line = line.strip()
        if not line:
            return None
        self.raw_logs.append(line)

        if line.startswith("[START]"):
            self.status = "RUNNING"
            self.progress_pct = 5.0
            return "START"

        elif line.startswith("[LAYER]"):
            # e.g., [LAYER] idx=1 name=Conv2d_1 progress=12.5%
            m_name = re.search(r"name=([A-Za-z0-9_]+)", line)
            m_prog = re.search(r"progress=([0-9.]+)%", line)
            if m_name:
                self.current_layer = m_name.group(1)
            if m_prog:
                self.progress_pct = float(m_prog.group(1))
            return "LAYER"

        elif line.startswith("[METRIC]"):
            # e.g., [METRIC] cycles=1420500 instrs=1120400 time_ms=14.20
            m_cyc = re.search(r"cycles=([0-9]+)", line)
            m_ins = re.search(r"instrs=([0-9]+)", line)
            m_tim = re.search(r"time_ms=([0-9.]+)", line)
            if m_cyc: self.cycles = int(m_cyc.group(1))
            if m_ins: self.instructions = int(m_ins.group(1))
            if m_tim: self.latency_ms = float(m_tim.group(1))
            if self.instructions > 0:
                self.cpi = self.cycles / self.instructions
            else:
                self.cpi = 1.27
            return "METRIC"


        elif line.startswith("[TOP5]"):
            # e.g., [TOP5] 0:airplane=-28 1:automobile=-270 3:cat=416 ...
            tokens = line.replace("[TOP5]", "").strip().split()
            scores = {}
            for tok in tokens:
                if "=" in tok:
                    cls_part, score_part = tok.split("=")
                    if ":" in cls_part:
                        _, cname = cls_part.split(":")
                    else:
                        cname = cls_part
                    scores[cname] = float(score_part)
            self.top5_scores = scores
            return "TOP5"

        elif line.startswith("[PRED]"):
            # e.g., [PRED] class_id=3 class_name=cat status=PASS
            m_id = re.search(r"class_id=([0-9]+)", line)
            m_name = re.search(r"class_name=([A-Za-z0-9_]+)", line)
            if m_id: self.pred_class_id = int(m_id.group(1))
            if m_name: self.pred_class_name = m_name.group(1)
            return "PRED"

        elif line.startswith("[DONE]"):
            self.status = "COMPLETED"
            self.progress_pct = 100.0
            return "DONE"

        return "LOG"


# --- Cached serial connection kept alive across runs ---
def get_serial_connection(port: str, baud_rate: int):
    """Open the port once per Streamlit session and reuse it."""
    if "serial_conn" in st.session_state:
        conn = st.session_state["serial_conn"]
        if conn.is_open and conn.port == port and conn.baudrate == baud_rate:
            return conn
        try:
            conn.close()
        except Exception:
            pass

    # Prevent DTR/RTS pulses on open (some CP210x + FPGA setups reset on this)
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud_rate
    ser.timeout = 2.0          # per-read timeout
    ser.dtr = False
    ser.rts = False
    ser.open()
    st.session_state["serial_conn"] = ser
    return ser


def _drain(ser, seconds=0.3):
    """Discard anything currently sitting in the RX buffer."""
    end = time.time() + seconds
    while time.time() < end:
        if ser.in_waiting:
            ser.read(ser.in_waiting)
            end = time.time() + 0.1
        else:
            time.sleep(0.02)


def _wait_for_idle(ser, parser, callback, timeout=5.0):
    """Wait until the board tells us it's IDLE (or timeout)."""
    ser.write(b"\x00")   # a byte the parser ignores — pokes it to print IDLE if silent
    end = time.time() + timeout
    while time.time() < end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parser.parse_line(line)
        if callback:
            callback(parser, line)
        if "[IDLE]" in line:
            return True
    return False


def stream_hardware_inference(port, baud_rate=115200, target_class="cat",
                              image_idx=0, custom_raw_bytes=None, callback=None):
    parser = VC707TelemetryParser()

    if port == "SIMULATION_MODE" or not HAS_SERIAL:
        # ... your existing simulation branch, unchanged ...
        return parser

    try:
        ser = get_serial_connection(port, baud_rate)

        # 1. Drain any leftover bytes from prior runs
        _drain(ser, 0.3)

        # 2. Send the command — NO newline, board triggers on the digit itself
        if custom_raw_bytes is not None and len(custom_raw_bytes) == 3072:
            ser.write(b"U")
            ser.flush()
            # Wait for [UPLOAD_READY] before dumping bytes
            waited = 0
            got_ready = False
            while waited < 3.0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    parser.parse_line(line)
                    if callback: callback(parser, line)
                    if "[UPLOAD_READY]" in line:
                        got_ready = True
                        break
                waited += 0.1
            if not got_ready:
                parser.raw_logs.append("[ERROR] Board did not enter UPLOAD_READY")
                return parser
            ser.write(custom_raw_bytes)
            ser.flush()
        else:
            ser.write(str(image_idx).encode("ascii"))   # just "0", not "0\n"
            ser.flush()

        # 3. Read until [DONE], with a watchdog on silence (not total time)
        SILENCE_TIMEOUT = 60.0       # kill if board goes quiet this long
        HARD_CAP        = 600.0      # absolute ceiling (10 min) as a safety net
        last_rx = time.time()
        start   = time.time()

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            now = time.time()
            if line:
                last_rx = now
                event = parser.parse_line(line)
                if callback:
                    callback(parser, line)
                if event == "DONE":
                    break
            else:
                if now - last_rx > SILENCE_TIMEOUT:
                    parser.raw_logs.append(
                        f"[ERROR] No UART data for {SILENCE_TIMEOUT}s — board hung?")
                    parser.status = "TIMEOUT"
                    break
                if now - start > HARD_CAP:
                    parser.raw_logs.append("[ERROR] Hard time cap hit.")
                    parser.status = "TIMEOUT"
                    break

    except Exception as e:
        parser.raw_logs.append(f"[ERROR] Serial Communication Failed: {e}")
        parser.status = "PORT_LOCKED_OR_OFFLINE"
        # If the port died, drop the cached handle so next click reopens it
        if "serial_conn" in st.session_state:
            try: st.session_state["serial_conn"].close()
            except Exception: pass
            del st.session_state["serial_conn"]

    return parser


