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
# import required libraries
from vidgear.gears import NetGear
import cv2
import numpy as np

# activate Multi-Clients mode
options = {
    "multiclient_mode": True,
    "max_retries": 10000,
    "jpeg_compression": False,
}

client = NetGear(
    address="127.0.0.1",
    port="5454",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

if __name__ == "__main__":
    while True:
        # receive data from server
        frame = client.recv()
        if frame is None:
            break
        rgb_frame = np.uint8(frame[:, :640])
        signal, disparity, depth = cv2.split(frame[:, 640:])
        signal, disparity = np.uint8(signal), np.uint8(disparity)

        if signal[0][0] == 255:
            cv2.putText(
                signal, "STOP!!!", (320, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 0
            )

        cv2.imshow("Depth-Client-Signal Display", signal)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    client.close()
