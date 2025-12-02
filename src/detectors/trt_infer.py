# src/detectors/trt_infer.py
import numpy as np
import cv2, time
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False

class TRTInfer:
    def __init__(self, engine_path, input_size=640):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT (tensorrt, pycuda) not available")
        self.engine_path = engine_path
        self.input_size = input_size
        self._load_engine()

    def _load_engine(self):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        import pycuda.driver as cuda
        self.stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            if self.engine.binding_is_input(name):
                self.inputs.append({'name':name,'host':host_mem,'device':dev_mem,'shape':shape,'dtype':dtype})
            else:
                self.outputs.append({'name':name,'host':host_mem,'device':dev_mem,'shape':shape,'dtype':dtype})
        print("[TRTInfer] loaded engine:", self.engine_path)

    def preprocess(self, frame):
        h0, w0 = frame.shape[:2]
        target = self.input_size
        r = min(target / w0, target / h0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = target - new_unpad[0], target - new_unpad[1]
        dw /= 2; dh /= 2
        resized = cv2.resize(frame, new_unpad)
        padded = cv2.copyMakeBorder(resized, int(round(dh)), int(round(dh)), int(round(dw)), int(round(dw)), cv2.BORDER_CONSTANT, value=(114,114,114))
        img = padded[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...].astype(np.float32)
        return img, r, (int(round(dw)), int(round(dh)))

    def infer(self, frame):
        # NOTE: This implementation assumes a single input and single output.
        img, r, pad = self.preprocess(frame)
        import pycuda.driver as cuda
        # copy to device
        np.copyto(np.asarray(self.inputs[0]['host']), img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # execute
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        # get output
        out = np.array(self.outputs[0]['host']).copy()
        # reshape according to output shape (best-effort)
        try:
            out = out.reshape(self.outputs[0]['shape'])
        except:
            pass
        # latency unknown precisely here
        return out, 0.001