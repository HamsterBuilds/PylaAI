import os
from functools import lru_cache
import time

import cv2
import numpy as np
import onnxruntime as ort
from utils import load_toml_as_dict
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning
)


@lru_cache(maxsize=8)
def _row_indices(length):
    """Reuse the fixed YOLO row index instead of allocating it every frame."""
    indices = np.arange(length)
    indices.flags.writeable = False
    return indices


def _numpy_nms(boxes, scores, iou_threshold=0.6, max_output=None):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = x2 - x1
    np.multiply(areas, y2 - y1, out=areas)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Suppression is greedy in score order: later iterations cannot
        # change this prefix. Stop once the caller has all returned boxes.
        if max_output is not None and len(keep) >= max_output:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[i], x1[remaining])
        yy1 = np.maximum(y1[i], y1[remaining])
        xx2 = np.minimum(x2[i], x2[remaining])
        yy2 = np.minimum(y2[i], y2[remaining])

        # Reuse coordinate temporaries for width, height and intersection.
        # The operation order matches the original IoU expression exactly.
        np.subtract(xx2, xx1, out=xx2)
        np.maximum(xx2, 0.0, out=xx2)
        np.subtract(yy2, yy1, out=yy2)
        np.maximum(yy2, 0.0, out=yy2)
        np.multiply(xx2, yy2, out=xx2)

        union = areas[remaining]
        np.add(union, areas[i], out=union)
        np.subtract(union, xx2, out=union)
        np.add(union, 1e-6, out=union)
        np.divide(xx2, union, out=xx2)
        order = remaining[xx2 <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def _normalize_yolo_output(raw_output):
    """
    Accepts either:
        outputs
        outputs[0]

    Supports common YOLO ONNX shapes:
        (1, 84, 8400)
        (1, 8400, 84)
        (84, 8400)
        (8400, 84)

    Returns:
        prediction with shape (num_boxes, num_channels)
    """

    if isinstance(raw_output, (list, tuple)):
        prediction = raw_output[0]
    else:
        prediction = raw_output

    prediction = np.asarray(prediction)

    if prediction.ndim == 3:
        prediction = prediction[0]

    if prediction.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {prediction.shape}")

    # YOLOv8 ONNX often gives (84, 8400), needs transpose to (8400, 84)
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 256:
        prediction = prediction.T

    return prediction


def _postprocess_raw(raw_output, conf_tresh=0.6, iou_thresh=0.6,
                     max_detections_per_class=64):
    prediction = _normalize_yolo_output(raw_output)

    n_detections = prediction.shape[0]
    n_classes = prediction.shape[1] - 4

    if n_classes <= 0:
        return []

    boxes_cxcywh = prediction[:, :4]
    class_scores = prediction[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[_row_indices(n_detections), class_ids]

    mask = confidences >= conf_tresh

    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Build xyxy directly. This produces the same values as four temporaries
    # plus np.stack, while avoiding those full-size per-frame arrays.
    boxes_xyxy = np.empty_like(boxes_cxcywh)
    # Use output columns 2/3 as temporary half-size buffers. This removes two
    # additional arrays from every entity and wall inference postprocess.
    np.multiply(boxes_cxcywh[:, 2], 0.5, out=boxes_xyxy[:, 2])
    np.multiply(boxes_cxcywh[:, 3], 0.5, out=boxes_xyxy[:, 3])
    np.subtract(boxes_cxcywh[:, 0], boxes_xyxy[:, 2],
                out=boxes_xyxy[:, 0])
    np.subtract(boxes_cxcywh[:, 1], boxes_xyxy[:, 3],
                out=boxes_xyxy[:, 1])
    np.add(boxes_cxcywh[:, 0], boxes_xyxy[:, 2],
           out=boxes_xyxy[:, 2])
    np.add(boxes_cxcywh[:, 1], boxes_xyxy[:, 3],
           out=boxes_xyxy[:, 3])

    results = []

    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls

        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]

        keep = _numpy_nms(cls_boxes, cls_scores, iou_thresh,
                          max_output=max_detections_per_class)
        keep = keep[:max_detections_per_class]

        if len(keep) == 0:
            continue

        # Assemble the final rows once instead of creating three intermediate
        # arrays and concatenating them. Column order and dtype stay identical.
        det = np.empty((len(keep), 6), dtype=np.float32)
        det[:, :4] = cls_boxes[keep]
        det[:, 4] = cls_scores[keep]
        det[:, 5] = cls

        results.append(det)

    return results


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None,
                 input_size=(640, 640), max_detections_per_class=64):
        threads_to_use = load_toml_as_dict("cfg/general_config.toml")['used_threads']

        def get_optimal_threads(max_limit=6):
            threads = os.cpu_count()
            threads_amount = min(max(2, threads // 2), max_limit)
            print(f"Detected {threads} CPU threads, using {threads_amount} threads.")
            return threads_amount

        self.optimal_threads_amount = get_optimal_threads() if threads_to_use == "auto" else int(threads_to_use)
        cv2.setNumThreads(self.optimal_threads_amount)
        self.preferred_device = str(
            load_toml_as_dict("cfg/general_config.toml").get("cpu_or_gpu", "auto") or "auto"
        ).strip().lower()
        self.model_path = model_path
        self.classes = classes
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.input_size = input_size
        self.max_detections_per_class = max(1, int(max_detections_per_class))
        self.model, self.device = self.load_model()
        if self.device in ("DmlExecutionProvider", "CUDAExecutionProvider"):
            # ONNX already occupies the GPU/driver and the emulator needs CPU
            # time concurrently. Four OpenCV workers for a single resize can
            # oversubscribe an 8-thread low-end host and create frame spikes;
            # two workers produce the identical pixels with steadier latency.
            cv2.setNumThreads(min(2, self.optimal_threads_amount))
        self.input_name = self.model.get_inputs()[0].name
        # Postprocessing intentionally consumes only the first model output.
        # Naming it avoids materializing unused outputs on multi-output exports.
        self.output_names = [self.model.get_outputs()[0].name]
        self._padded_img_buffer = np.full(
            (1, 3, self.input_size[0], self.input_size[1]),
            128.0 / 255.0,
            dtype=np.float32
        )
        self._normalization_scale = np.float32(1.0 / 255.0)
        self._input_feed = {self.input_name: self._padded_img_buffer}
        self._preprocess_source_shape = None
        self._preprocess_target = None
        self._io_binding = None
        self._output_buffer = None
        self.total_detection_seconds = 0.0
        self.detection_calls = 0
        self.maximum_detection_seconds = 0.0
        self._configure_static_io_binding()

    def _configure_static_io_binding(self):
        """Bind static host output once so inference can reuse its allocation."""
        # DirectML's transfer into a caller-owned CPU buffer is provider- and
        # driver-sensitive. Keep its established Session.run path; a stale
        # bound output makes valid frames look as if they contain no entities.
        if self.device == "DmlExecutionProvider":
            return
        output = self.model.get_outputs()[0]
        shape = tuple(output.shape)
        if (
            output.type != "tensor(float)"
            or not shape
            or any(not isinstance(size, int) or size <= 0 for size in shape)
        ):
            return
        try:
            output_buffer = np.empty(shape, dtype=np.float32)
            io_binding = self.model.io_binding()
            io_binding.bind_cpu_input(
                self.input_name, self._padded_img_buffer
            )
            io_binding.bind_output(
                self.output_names[0],
                device_type="cpu",
                device_id=0,
                element_type=np.float32,
                shape=shape,
                buffer_ptr=output_buffer.ctypes.data,
            )
            self._output_buffer = output_buffer
            self._io_binding = io_binding
        except Exception:
            # Dynamic/older runtimes retain the standard allocation path.
            self._output_buffer = None
            self._io_binding = None

    def load_model(self):
        available_providers = ort.get_available_providers()
        providers = []

        if self.preferred_device in ("gpu", "auto"):
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")

        providers.append("CPUExecutionProvider")
        if self.preferred_device == "cpu":
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = self.optimal_threads_amount
        so.inter_op_num_threads = 1
        # Separate detector pools must not busy-wait while the emulator or
        # another detector is using this small CPU.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        model = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)

        used_provider = model.get_providers()[0]
        if used_provider == "CUDAExecutionProvider":
            print("Using CUDA GPU")
        elif used_provider == "DmlExecutionProvider":
            print("Using GPU")
        elif self.preferred_device != "cpu":
            print("Using CPU as no GPU provider found")

        return model, used_provider

    def _recover_on_cpu(self, error):
        """Replace a failed GPU session instead of terminating the bot."""
        if self.device == "CPUExecutionProvider":
            raise error
        print(
            f"{self.device} inference failed; switching this detector to CPU: "
            f"{error}"
        )
        self.preferred_device = "cpu"
        self.model, self.device = self.load_model()
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [self.model.get_outputs()[0].name]
        self._input_feed = {self.input_name: self._padded_img_buffer}
        self._io_binding = None
        self._output_buffer = None

    def preprocess_image(self, img):
        h, w = img.shape[:2]

        source_shape = (h, w)
        if self._preprocess_source_shape != source_shape:
            scale = min(self.input_size[0] / h, self.input_size[1] / w)
            self._new_w = int(w * scale)
            self._new_h = int(h * scale)
            self._preprocess_source_shape = source_shape
            self._resize_buffer = np.empty(
                (self._new_h, self._new_w, 3), dtype=np.uint8
            )
            self._preprocess_target = self._padded_img_buffer[
                0, :, :self._new_h, :self._new_w
            ]
            # A changed aspect ratio must not leave old image pixels in padding.
            self._padded_img_buffer.fill(128.0 / 255.0)
        resized_img = cv2.resize(
            img,
            (self._new_w, self._new_h),
            dst=self._resize_buffer,
            interpolation=cv2.INTER_LINEAR
        )

        np.multiply(
            resized_img.transpose(2, 0, 1), self._normalization_scale,
            out=self._preprocess_target,
        )

        return self._padded_img_buffer, self._new_w, self._new_h

    def postprocess(self, raw_output, orig_img_shape, resized_shape, conf_tresh=0.6):
        detections = _postprocess_raw(
            raw_output,
            conf_tresh=conf_tresh,
            iou_thresh=0.6,
            max_detections_per_class=self.max_detections_per_class,
        )

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape

        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []

        for det in detections:
            if len(det):
                det[:, 0] *= scale_w
                det[:, 1] *= scale_h
                det[:, 2] *= scale_w
                det[:, 3] *= scale_h
                np.clip(det[:, 0], 0, orig_w - 1, out=det[:, 0])
                np.clip(det[:, 2], 0, orig_w - 1, out=det[:, 2])
                np.clip(det[:, 1], 0, orig_h - 1, out=det[:, 1])
                np.clip(det[:, 3], 0, orig_h - 1, out=det[:, 3])
                results.append(det)

        return results

    def detect_objects(self, img, conf_tresh=0.6):
        detection_started = time.perf_counter()
        orig_h, orig_w = img.shape[:2]

        _, resized_w, resized_h = self.preprocess_image(img)

        if self._io_binding is not None:
            try:
                self.model.run_with_iobinding(self._io_binding)
                # DirectML may complete a pre-bound CPU output transfer
                # asynchronously. Do not let NumPy postprocessing observe the
                # previous/unfinished buffer contents.
                self._io_binding.synchronize_outputs()
                outputs = (self._output_buffer,)
            except Exception as error:
                # Disable a binding permanently after a runtime/provider
                # rejection and complete this frame through the proven path.
                print(f"Reusable ONNX output disabled: {error}")
                self._io_binding = None
                self._output_buffer = None
                outputs = self.model.run(
                    self.output_names, self._input_feed
                )
        else:
            try:
                outputs = self.model.run(
                    self.output_names, self._input_feed
                )
            except Exception as error:
                self._recover_on_cpu(error)
                outputs = self.model.run(
                    self.output_names, self._input_feed
                )

        detections = self.postprocess(
            outputs,
            (orig_h, orig_w),
            (resized_w, resized_h),
            conf_tresh
        )

        results = {}
        classes = self.classes
        ignore_classes = self.ignore_classes
        get_result = results.get

        for detection in detections:
            if not len(detection):
                continue
            coordinates = detection[:, :4].astype(
                np.int32, copy=False
            ).tolist()
            # Each postprocess block was created from exactly one class, so
            # resolve and filter it once instead of once per detected box.
            class_id = int(detection[0, 5])
            if classes is None:
                class_name = str(class_id)
            else:
                if class_id < 0 or class_id >= len(classes):
                    print(
                        f"WARNING: class_id {class_id} is out of range "
                        f"(classes length: {len(classes)}). Detection ignored."
                    )
                    continue
                class_name = classes[class_id]

            if class_id in ignore_classes or class_name in ignore_classes:
                continue

            class_results = get_result(class_name)
            if class_results is None:
                results[class_name] = coordinates
            else:
                class_results.extend(coordinates)

        elapsed = time.perf_counter() - detection_started
        self.total_detection_seconds += elapsed
        self.detection_calls += 1
        if elapsed > self.maximum_detection_seconds:
            self.maximum_detection_seconds = elapsed
        return results

    def detection_timing_ms(self):
        if not self.detection_calls:
            return 0.0, 0.0
        return (
            self.total_detection_seconds * 1000.0 / self.detection_calls,
            self.maximum_detection_seconds * 1000.0,
        )
