# -*- coding: utf-8 -*-
# Universal TensorRT execute_async_v3 runner (Dynamic/Static auto, Warm-up included)
import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class TensorRTRunner:
    """Universal TensorRT runner with execute_async_v3 and auto dynamic/static handling."""

    def __init__(self, onnx_path, engine_path=None, fp16=False, int8=False, input_profiles=None):
        self.onnx_path = onnx_path
        self.engine_path = engine_path or onnx_path.replace(".onnx", ".engine")
        self.fp16 = fp16
        self.int8 = int8
        self.input_profiles = input_profiles

        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()

        self.bindings = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.is_dynamic = False

    # ---------------- Build or Load ----------------
    def build_or_load_engine(self):
        if os.path.exists(self.engine_path):
            with open(self.engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        else:
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(flags=trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            parser = trt.OnnxParser(network, TRT_LOGGER)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._GiB(2))

            timing_cache = os.path.splitext(self.engine_path)[0] + "_timing.cache"
            self._setup_timing_cache(config, timing_cache)

            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

            if self.input_profiles:
                profile = builder.create_optimization_profile()
                for name, (min_s, opt_s, max_s) in self.input_profiles.items():
                    profile.set_shape(name, min_s, opt_s, max_s)
                config.add_optimization_profile(profile)

            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(f"ONNX not found: {self.onnx_path}")
            parser.parse_from_file(self.onnx_path)

            plan = builder.build_serialized_network(network, config)
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            self._save_timing_cache(config, timing_cache)
            with open(self.engine_path, "wb") as f:
                f.write(plan)

        self.context = self.engine.create_execution_context()
        self._analyze_bindings()
        self._warmup()

    # ---------------- GiB ----------------
    def _GiB(self, val):
        return val * (1 << 30)
    
    # ---------------- setup timing cache ----------------
    def _setup_timing_cache(self, config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
        buffer = b""
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                buffer = f.read()
        cache = config.create_timing_cache(buffer)
        config.set_timing_cache(cache, True)

    # ---------------- save timing cache ----------------
    def _save_timing_cache(self, config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
        cache = config.get_timing_cache()
        with open(timing_cache_path, "wb") as f:
            f.write(memoryview(cache.serialize()))

    # ---------------- Binding Analysis ----------------
    def _analyze_bindings(self):
        self.bindings.clear()
        if hasattr(self.engine, "num_io_tensors"):  # TRT 9.x
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                is_input = mode == trt.TensorIOMode.INPUT
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.engine.get_tensor_shape(name))
                self.bindings[name] = {"index": i, "name": name, "is_input": is_input, "dtype": dtype, "shape": shape}
                print(self.bindings[name])
        else:  # TRT 8.x
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = tuple(self.engine.get_binding_shape(i))
                self.bindings[name] = {"index": i, "name": name, "is_input": is_input, "dtype": dtype, "shape": shape}
                print(self.bindings[name])

        self.is_dynamic = any(-1 in v["shape"] for v in self.bindings.values())

    # ---------------- Buffer Management ----------------
    def _alloc_device(self, shape, dtype):
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        return cuda.mem_alloc(nbytes)

    def _ensure_buffer(self, name, shape, dtype, is_input):
        buf = self.input_buffers if is_input else self.output_buffers
        if name not in buf or buf[name][1] != shape:
            if name in buf:
                buf[name][0].free()
            dev = self._alloc_device(shape, dtype)
            buf[name] = (dev, shape, dtype)
        return buf[name][0]

    # ---------------- Shape Getter ----------------
    def _get_shape(self, name, info):
        if hasattr(self.context, "get_tensor_shape"):  # TRT 9.x
            return tuple(self.context.get_tensor_shape(name))
        return tuple(self.context.get_binding_shape(info["index"]))  # TRT 8.x

    # ---------------- Warm-up ----------------
    def _warmup(self, warmup_iters=30):
        """Run dummy inference to initialize buffers and kernels."""
        print(f"[TRT] Warm-up ({'dynamic' if self.is_dynamic else 'static'}) start...")
        dummy_inputs = {}
        for name, info in self.bindings.items():
            if info["is_input"]:
                shape = info["shape"]
                if self.is_dynamic:
                    shape = tuple(s if s > 0 else 1 for s in shape)
                arr = np.random.rand(*shape).astype(info["dtype"])
                dummy_inputs[name] = arr
        for _ in range(warmup_iters):
            _ = self.infer(dummy_inputs, return_device=False)
        print("[TRT] Warm-up complete.")

    # ---------------- Inference ----------------
    def infer(self, inputs, return_device=False):
        if self.engine is None or self.context is None:
            raise RuntimeError("Call build_or_load_engine() first")

        # Dynamic shape 설정
        for name, arr in inputs.items():
            info = self.bindings[name]
            if self.is_dynamic:
                if hasattr(self.context, "set_input_shape"):
                    self.context.set_input_shape(name, arr.shape)
                else:
                    self.context.set_binding_shape(info["index"], arr.shape)

        # 입력 업로드
        for name, arr in inputs.items():
            info = self.bindings[name]
            np_arr = np.ascontiguousarray(arr.astype(info["dtype"]))
            dev_ptr = self._ensure_buffer(name, np_arr.shape, np_arr.dtype, True)
            cuda.memcpy_htod_async(dev_ptr, np_arr, self.stream)
            self.context.set_tensor_address(name, int(dev_ptr))

        # 출력 준비
        output_info = {}
        for name, info in self.bindings.items():
            if info["is_input"]:
                continue
            out_shape = self._get_shape(name, info)
            dev_ptr = self._ensure_buffer(name, out_shape, info["dtype"], False)
            self.context.set_tensor_address(name, int(dev_ptr))
            output_info[name] = (dev_ptr, out_shape, info["dtype"])

        # 실행
        t0 = time.time()
        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()
        t1 = time.time()

        # 결과 복사
        outputs = {"infer_time_ms": (t1 - t0) * 1000.0}
        for name, (dev_ptr, shape, dtype) in output_info.items():
            if return_device:
                outputs[name] = dev_ptr
            else:
                host = np.empty(shape, dtype=dtype)
                cuda.memcpy_dtoh(host, dev_ptr)
                outputs[name] = host
        return outputs

    # ---------------- Cleanup ----------------
    def destroy(self):
        for buf in [self.input_buffers, self.output_buffers]:
            for dev, _, _ in buf.values():
                try:
                    dev.free()
                except:
                    pass
            buf.clear()
        del self.context, self.engine
