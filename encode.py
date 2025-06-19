import depthai as dai
import threading
import signal
import time

PROFILE = dai.VideoEncoderProperties.Profile.H265_MAIN
RECORD_SECONDS = 20

quitEvent = threading.Event()
signal.signal(signal.SIGTERM, lambda *_args: quitEvent.set())
signal.signal(signal.SIGINT, lambda *_args: quitEvent.set())

class FrameCounterSaver(dai.node.HostNode):
    def __init__(self, *args, **kwargs):
        dai.node.HostNode.__init__(self, *args, **kwargs)
        self.file_handle = open('video.encoded', 'wb')
        self.frame_count = 0

    def build(self, *args):
        self.link_args(*args)
        return self

    def process(self, frame):
        self.frame_count += 1
        frame.getData().tofile(self.file_handle)

device = dai.Device(dai.UsbSpeed.HIGH)  # Use HIGH for USB2.0, SUPER for USB3.0

with dai.Pipeline(device) as pipeline:
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    encoded = pipeline.create(dai.node.VideoEncoder).build(
        camRgb.requestOutput((1920, 1080), type=dai.ImgFrame.Type.NV12),
        frameRate=30,
        profile=PROFILE
    )
    saver = pipeline.create(FrameCounterSaver).build(encoded.out)

    pipeline.start()
    print(f"Recording for {RECORD_SECONDS} seconds...")

    start_time = time.monotonic()
    while pipeline.isRunning() and not quitEvent.is_set():
        elapsed = time.monotonic() - start_time
        if elapsed >= RECORD_SECONDS:
            break
        time.sleep(0.1)  # sleep to reduce CPU usage

    duration = time.monotonic() - start_time
    frame_count = saver.frame_count
    pipeline.stop()
    pipeline.wait()
    saver.file_handle.close()

    fps = frame_count / duration if duration > 0 else 0
    print(f"\n--- Recording stats ---")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frames encoded: {frame_count}")
    print(f"Effective FPS: {fps:.2f}")
    print("\nTo convert video.encoded to .mp4, run:")
    print("ffmpeg -framerate 30 -i video.encoded -c copy video.mp4")
