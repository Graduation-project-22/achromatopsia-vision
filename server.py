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
import time
import depthai as dai
from vidgear.gears import NetGear
import cv2
import numpy as np

# NetGear Server Definition
options = {
    "multiclient_mode": True,
    "max_retries": 10000,
    "jpeg_compression": False,
}

server = NetGear(
    address="127.0.0.1",
    port=(
        5454,
        5455,
    ),
    protocol="tcp",
    pattern=1,
    logging=True,
    **options,
)

SYNC_NN = True
NN_BLOB_PATH = "mobilenet-ssd_openvino_2021.4_6shave.blob"
LABEL_MAP = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class CameraDefinition:
    def __init__(self) -> None:
        self.pipeline = dai.Pipeline()
        self.cam_rgb = None
        self.mono_right = None
        self.mono_left = None
        self.stereo = None
        self.spatial_detection_network = None
        self.xout_rgb = None
        self.xout_depth = None
        self.xout_desparity = None
        self.xout_nn = None
        self.xout_bounding_box_depth_mapping = None

        self.camera_rgb_node()
        self.mono_right_node()
        self.mono_left_node()
        self.stereo_node()
        self.spatial_detection_network_node()
        self.xout_rgb_node()
        self.xout_stereo_node()
        self.xout_spatial_detection_network_node()

    def camera_rgb_node(self) -> None:
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        self.cam_rgb.setPreviewSize(300, 300)
        self.cam_rgb.setInterleaved(False)

    def mono_right_node(self) -> None:
        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        self.mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        self.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    def mono_left_node(self) -> None:
        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        self.mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        self.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    def stereo_node(self) -> None:
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )

    def spatial_detection_network_node(self) -> None:
        self.spatial_detection_network = self.pipeline.create(
            dai.node.MobileNetSpatialDetectionNetwork
        )
        self.spatial_detection_network.setBlobPath(NN_BLOB_PATH)
        self.spatial_detection_network.setConfidenceThreshold(0.5)
        self.spatial_detection_network.input.setBlocking(False)
        self.spatial_detection_network.setBoundingBoxScaleFactor(0.5)
        self.spatial_detection_network.setDepthLowerThreshold(100)
        self.spatial_detection_network.setDepthUpperThreshold(5000)
        self.cam_rgb.preview.link(self.spatial_detection_network.input)
        self.stereo.depth.link(self.spatial_detection_network.inputDepth)

    def xout_rgb_node(self) -> None:
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        if SYNC_NN:
            self.spatial_detection_network.passthrough.link(
                self.xout_rgb.input
            )
        else:
            self.cam_rgb.preview.link(self.xout_rgb.input)

    def xout_stereo_node(self) -> None:
        self.mono_left.out.link(self.stereo.left)
        self.mono_right.out.link(self.stereo.right)

        self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
        self.xout_depth.setStreamName("depth")
        if SYNC_NN:
            self.spatial_detection_network.passthroughDepth.link(
                self.xout_depth.input
            )
        else:
            self.stereo.depth.link(self.xout_depth.input)

        self.xout_disparity = self.pipeline.create(dai.node.XLinkOut)
        self.xout_disparity.setStreamName("disparity")
        self.stereo.disparity.link(self.xout_disparity.input)

    def xout_spatial_detection_network_node(self) -> None:
        self.xout_nn = self.pipeline.create(dai.node.XLinkOut)
        self.xout_nn.setStreamName("detections")
        self.spatial_detection_network.out.link(self.xout_nn.input)

        self.xout_bounding_box_depth_mapping = self.pipeline.create(
            dai.node.XLinkOut
        )
        self.xout_bounding_box_depth_mapping.setStreamName("BBDMapping")
        self.spatial_detection_network.boundingBoxMapping.link(
            self.xout_bounding_box_depth_mapping.input
        )

    def get_max_disparity(self) -> None:
        return self.stereo.initialConfig.getMaxDisparity()

    def get_pipeline(self) -> None:
        return self.pipeline


oak = CameraDefinition()
pipeline = oak.get_pipeline()
max_disparity = oak.get_max_disparity()
startTime = time.monotonic()
counter = 0
fps = 0

if __name__ == "__main__":
    with dai.Device(pipeline) as device:
        print("Connected cameras: ", device.getConnectedCameras())
        print("Usb speed: ", device.getUsbSpeed().name)

        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_disparity = device.getOutputQueue(
            name="disparity", maxSize=4, blocking=False
        )
        q_depth = device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        detection_nn_queue = device.getOutputQueue(
            name="detections", maxSize=4, blocking=False
        )
        xout_bounding_box_depth_mapping = device.getOutputQueue(
            name="BBDMapping", maxSize=4, blocking=False
        )
        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()
            in_disparity = q_disparity.get()
            in_detections = detection_nn_queue.get()

            rgb_frame_model = in_rgb.getCvFrame()
            rgb_frame = cv2.resize(rgb_frame_model.copy(), (640, 400))
            disparity_frame = np.uint16(
                (in_disparity.getFrame() * (255 / max_disparity)).astype(
                    np.uint8
                )
            )
            depth_frame = in_depth.getFrame()
            detections = in_detections.detections

            height = rgb_frame_model.shape[0]
            width = rgb_frame_model.shape[1]

            signal = np.zeros((400, 640), dtype=np.uint8)
            for detection in detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                try:
                    label = LABEL_MAP[detection.label]
                except IndexError:
                    label = detection.label

                if label in ["person"]:
                    distance = np.sqrt(
                        (int(detection.spatialCoordinates.x)) ** 2
                        + (int(detection.spatialCoordinates.y)) ** 2
                        + (int(detection.spatialCoordinates.z)) ** 2
                    )

                    if distance <= 500:
                        signal[:] = 255

                    # for demonstration
                    cv2.putText(
                        rgb_frame_model,
                        str(label),
                        (x1 + 10, y1 + 20),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        rgb_frame_model,
                        "{:.2f}".format(detection.confidence * 100),
                        (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        rgb_frame_model,
                        f"X: {int(detection.spatialCoordinates.x)} mm",
                        (x1 + 10, y1 + 50),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        rgb_frame_model,
                        f"Y: {int(detection.spatialCoordinates.y)} mm",
                        (x1 + 10, y1 + 65),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        rgb_frame_model,
                        f"Z: {int(detection.spatialCoordinates.z)} mm",
                        (x1 + 10, y1 + 80),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        rgb_frame_model,
                        "NN fps: {:.2f}".format(fps),
                        (2, rgb_frame_model.shape[0] - 4),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.4,
                        (255, 255, 255),
                    )
                    cv2.rectangle(
                        rgb_frame_model,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

            merged_channels = cv2.merge(
                [np.uint16(signal), disparity_frame, depth_frame]
            )

            combined_frame = np.hstack([np.uint16(rgb_frame), merged_channels])

            server.send(combined_frame)

            # FPS Calculation
            counter += 1
            current_time = time.monotonic()
            if (current_time - startTime) > 1:
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            cv2.imshow("frame", rgb_frame_model)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        server.close()
