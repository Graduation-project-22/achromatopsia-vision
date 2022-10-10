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
from queue import Queue
from vidgear.gears import NetGear
import cv2
import numpy as np
from trt_model import TRTInference


class myThread(threading.Thread):
    def __init__(self, func, args, model_output_queue):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.model_output_queue = model_output_queue

    def run(self):
        self.func(*self.args, self.model_output_queue)


# activate Multi-Clients mode
options = {
    "multiclient_mode": True,
    "max_retries": 10000,
    "jpeg_compression": False,
}

client = NetGear(
    address="127.0.0.1",
    port="5455",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

if __name__ == "__main__":
    TRT_ENGINE_PATH = "color_correction.trt"
    
    trt_inference_wrapper = TRTInference(
        TRT_ENGINE_PATH,
    )

    GO_TO_THREAD = True
    model_output_queue = Queue()
    color_corrected_image = np.zeros((256, 512, 3), dtype=np.uint8)
    while True:
        frame = client.recv()
        if frame is None:
            break
        # model_input = np.uint8(frame[:, :640])
        model_input = cv2.imread("val/{}.jpg".format(np.random.randint(1, 20)))[:, 256:]

        if GO_TO_THREAD:
            GO_TO_THREAD = False
            thread1 = myThread(
                trt_inference_wrapper.infer, [model_input], model_output_queue
            )
            thread1.start()
        else:
            if not model_output_queue.empty():
                thread1.join()
                GO_TO_THREAD = True
                output = model_output_queue.get()
                output = output.reshape(1, 3, 256, 256)
                output = np.squeeze(output)
                output = (
                    output * 0.5 + 0.5
                )  # handing output from tanh activation functions
                result = np.transpose(output.copy(), (1, 2, 0))
                result = cv2.cvtColor(
                    (result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                color_corrected_image = np.hstack(
                    [cv2.resize(model_input, (256, 256)), result]
                )
        cv2.imshow("color_corrected_image", color_corrected_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    client.close()
