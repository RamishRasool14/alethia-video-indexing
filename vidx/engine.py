"""YOLO11l inference backends returning the raw (B, 84, N) prediction tensor.

- TensorRT FP16 engine (built once by prepare.py on the target GPU), with a
  dynamic profile for batch and image size, optimised for 384x640.
- PyTorch fallback (Ultralytics model, FP16) if the engine is missing or
  TensorRT cannot load it.
"""
import os

import torch

MAX_BATCH = 64
OPT_HW = (384, 640)


def export_onnx(weights, onnx_path):
    from ultralytics import YOLO

    out = YOLO(weights).export(format="onnx", dynamic=True, imgsz=640, simplify=True,
                               opset=17, verbose=False)
    if os.path.abspath(out) != os.path.abspath(onnx_path):
        os.replace(out, onnx_path)
    return onnx_path


class _Calibrator:
    """INT8 entropy calibrator fed with letterboxed uint8 frames (N, H, W, 3) BGR."""

    def __new__(cls, frames, cache_path, batch=16):
        import tensorrt as trt

        class Calib(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.frames, self.batch, self.i = frames, batch, 0
                self.cache_path = cache_path
                self.buf = None

            def get_batch_size(self):
                return self.batch

            def get_batch(self, names):
                if self.i + self.batch > len(self.frames):
                    return None
                x = torch.from_numpy(self.frames[self.i:self.i + self.batch]).cuda()
                self.buf = x.permute(0, 3, 1, 2).flip(1).float().div(255).contiguous()
                self.i += self.batch
                return [int(self.buf.data_ptr())]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_path):
                    return open(self.cache_path, "rb").read()
                return None

            def write_calibration_cache(self, cache):
                open(self.cache_path, "wb").write(cache)

        return Calib()


def build_engine(onnx_path, engine_path, max_batch=MAX_BATCH, opt_hw=OPT_HW,
                 int8_frames=None, int8_cache=None, fp16_io=False, opt_level=None):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + "; ".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if opt_level is not None:
        config.builder_optimization_level = opt_level
    inp = network.get_input(0)
    if fp16_io:  # feed half precision directly (pre-processing writes FP16)
        inp.dtype = trt.float16
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, 3, 32, 32), (max_batch, 3, *opt_hw),
                      (max_batch, 3, 640, 640))
    config.add_optimization_profile(profile)
    if int8_frames is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        h, w = int8_frames.shape[1:3]
        calib_profile = builder.create_optimization_profile()
        calib_profile.set_shape(inp.name, (16, 3, h, w), (16, 3, h, w), (16, 3, h, w))
        config.set_calibration_profile(calib_profile)
        config.int8_calibrator = _Calibrator(int8_frames, int8_cache)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(data)
    return engine_path


class TRTModel:
    """Loaded engine; call runner() once per inference thread."""

    def __init__(self, engine_path, device="cuda:0"):
        import tensorrt as trt

        self.trt = trt
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("could not deserialize TensorRT engine")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = torch.float16 if self.engine.get_tensor_dtype(self.inp) == trt.float16 \
            else torch.float32
        self.out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.out) == trt.float16 \
            else torch.float32
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self._default = None

    def runner(self):
        """A callable with its own execution context, run on the caller's stream."""
        ctx = self.engine.create_execution_context()

        def run(x):
            x = x.to(self.in_dtype).contiguous()
            ctx.set_input_shape(self.inp, tuple(x.shape))
            out = torch.empty(tuple(ctx.get_tensor_shape(self.out)), dtype=self.out_dtype,
                              device=self.device)
            ctx.set_tensor_address(self.inp, x.data_ptr())
            ctx.set_tensor_address(self.out, out.data_ptr())
            ctx.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
            return out

        return run

    def __call__(self, x):
        if self._default is None:
            self._default = self.runner()
        return self._default(x)


class TorchModel:
    def __init__(self, weights, device="cuda:0", half=True):
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.half = half
        self.in_dtype = torch.float16 if half else torch.float32
        m = YOLO(weights).model.to(self.device).eval()
        m = m.fuse() if hasattr(m, "fuse") else m
        self.model = m.half() if half else m.float()
        self.max_batch = MAX_BATCH

    @torch.inference_mode()
    def __call__(self, x):
        x = x.half() if self.half else x.float()
        y = self.model(x)
        return y[0] if isinstance(y, (list, tuple)) else y

    def runner(self):
        return self


def load_model(weights, engine_path, log=print):
    if engine_path and os.path.exists(engine_path):
        try:
            m = TRTModel(engine_path)
            log(f"backend: TensorRT engine {engine_path}")
            return m
        except Exception as e:  # fall back rather than fail the run
            log(f"TensorRT unavailable ({type(e).__name__}: {e}); using PyTorch FP16")
    else:
        log("backend: PyTorch FP16 (no TensorRT engine found)")
    return TorchModel(weights)
