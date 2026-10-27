"""
.binetic format — the model packaging and inference protocol.

Ships:
- Quantum/ternary weights (quantized)
- 4D metadata per weight (address, phase+, phase-, time)
- Phase-residual configuration
- Custom protocol + ISA (instruction set)

A binary format that allows a CPU-based, phase-triggered inference engine.
This is the "universal accelerator" — the same states processed on
phase-lock hardware eliminate the need for GPU compute lattices.
"""
import struct
import os
import numpy as np
from enum import Enum


class BineticVersion:
    V4 = 4


class OpType(Enum):
    EXPERIENCE = 0x01
    LEARNING = 0x02
    MEMORY = 0x03
    TOOL = 0x04
    REASONING = 0x05
    GATE = 0x06
    MERGE = 0x07


class BineticHeader:
    """64-byte header for the .binetic format."""
    MAGIC = b'BNET'
    SIZE = 64

    def __init__(self, version=4, opset=None, flags=0, next_offset=0):
        self.version = version
        self.opset = opset or 0
        self.flags = flags
        self.next_offset = next_offset

    def pack(self):
        magic = b'BNET'
        version = struct.pack('<I', self.version)  # file format
        opset = struct.pack('<I', self.opset)     # max opcode
        flags = struct.pack('<I', self.flags)     # reserved
        next_offset = struct.pack('<Q', self.next_offset)  # next section
        return magic + version + opset + flags + next_offset


class BineticInstruction:
    """One op: (addr←reg, addr→reg, size, bit_width, phase, tick)."""

    def __init__(self, op_code=0x00, dst_addr=0, src_addr=0, size=1, bit_width=2, phase=0.0, tick=0):
        self.op_code = op_code
        self.dst_addr = dst_addr
        self.src_addr = src_addr
        self.size = size
        self.bit_width = bit_width
        self.phase = phase
        self.tick = tick

    def pack(self):
        return struct.pack('<QQQIIdsf',
                           self.op_code, self.dst_addr, self.src_addr,
                           self.size, self.bit_width, self.phase, self.tick)


class BineticWriter:
    """Builds a .binetic file: header + metadata + tensors + instructions."""

    def __init__(self, path):
        self.path = path
        self.instructions = []
        self.metadata = {}

    def add_instruction(self, instruction: BineticInstruction):
        self.instructions.append(instruction)

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def add_tensor(self, name, data, byte_width=2):
        """Add a quantized tensor with its 4D metadata."""
        self.metadata[name] = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'byte_width': byte_width,
        }

    def write(self):
        path = self.path
        with open(path, 'wb') as f:
            # Header (64 bytes)
            header = BineticHeader()
            f.write(header.pack())

            # Metadata
            f.write(struct.pack('<Q', len(self.metadata)))
            for key, value in self.metadata.items():
                f.write(struct.pack('<Q', len(key)))
                f.write(key.encode())
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(struct.pack('<Q', len(k)))
                        f.write(k.encode())
                        f.write(struct.pack('<Q', len(v)))
                        f.write(v.encode() if isinstance(v, str) else b'')

            # Instructions
            f.write(struct.pack('<Q', len(self.instructions)))
            for instruction in self.instructions:
                f.write(instruction.pack())

            # Tensors (quad-form)
            for name, meta in self.metadata.items():
                if 'data' not in meta:
                    continue
                # Pack the tensor
                data = meta['data']
                f.write(struct.pack('<Q', len(name)))
                f.write(name.encode())
                f.write(struct.pack('<I', data.shape[0]))
                f.write(struct.pack('<I', data.shape[1]))
                f.write(struct.pack('<B', meta.get('byte_width', 2)))
                f.write(data.tobytes())

        print(f"[binetic] wrote {path} ({os.path.getsize(path)} bytes)")


class BineticReader:
    """Reads a .binetic file, reconstructs the model + ISA."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.metadata = {}
        self.instructions = []
        self.tensors = {}

    def read(self):
        with open(self.path, 'rb') as f:
            # Header
            self.header = struct.unpack('<4sIIIQ', f.read(BineticHeader.SIZE))

            # Metadata
            n_meta = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n_meta):
                k_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(k_len).decode()
                self.metadata[key] = f.read(n if n > 0 else 100)  # read until break

        print(f"[binetic] read {self.path}: {len(self.metadata)} metadata, {len(self.instructions)} instructions")
