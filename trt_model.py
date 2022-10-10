"""
Copyright 2022 Achromatopsia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import threading
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


class TRTInference:
    def __init__(self, trt_engine_path):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, "rb") as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = (
                trt.volume(engine.get_binding_shape(binding))
                * engine.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, np.float16)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def image_loader(self, image):
        loader = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )  # (3, 256, 256) -> (1, 3, 256, 256)
        return loader(image).unsqueeze(0)

    def get_input(self, cv_img):
        original = cv_img.copy()
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        input = self.image_loader(Image.fromarray(cv_img))
        return original, np.array(input, dtype=np.float16)

    def infer(self, input_img_path, imagesQ):
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream = self.stream
        context = self.context

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # read image
        image = self.get_input(input_img_path)[1]
        np.copyto(host_inputs[0], image.ravel())

        # inference
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()

        # parse output
        self.cfx.pop()
        imagesQ.put(host_outputs[0])

    def destory(self):
        self.cfx.pop()
