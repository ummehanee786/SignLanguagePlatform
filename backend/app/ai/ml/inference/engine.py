"""
engine.py

This is the ONE file the rest of the application should ever need to
import from. Everything else in app/ai/ml/inference/ and app/ai/handtracking/
is an internal implementation detail.

    from app.ai.ml.inference.engine import predict
    result = predict(image)   # image = a BGR frame, e.g. from cv2.imread
                                # or a decoded webcam upload

`result` is always a PredictionResult (see result.py) - predict() never
raises for expected, everyday failure cases (no hand in frame, bad
landmarks, corrupt image). Callers check `result.success` instead of
wrapping every call in try/except.

Full pipeline implemented here:

    Raw Image
      -> MediaPipe Hand Detection      (HandLandmarkDetector)
      -> 21 Hand Landmarks
      -> Landmark Validation           (hand count: reject 0 or >1 hands;
                                         feature_pipeline.validate_landmarks
                                         for value/shape correctness)
      -> Feature Normalization         (feature_pipeline.build_feature_vector)
      -> 63-Dimensional Feature Vector
      -> Model Loading                 (model_loader.load_model, cached, latest version)
      -> Prediction
      -> Probability Calculation       (model.predict_proba)
      -> Confidence Threshold
      -> Prediction Object             (PredictionResult)
"""

import logging
import time
from typing import Optional

from app.ai.handtracking.detector import HandLandmarkDetector, decode_image
from app.ai.ml.inference.feature_pipeline import build_feature_vector, FeatureValidationError
from app.ai.ml.inference.model_loader import load_model, ModelNotFoundError, ModelVersionError
from app.ai.ml.inference.result import PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# The trained model only knows single-hand ASL alphabet gestures - a frame
# with more than one hand is out of scope for what this model can
# meaningfully classify, so it's rejected explicitly rather than silently
# picking one hand and ignoring the other.
MAX_SUPPORTED_HANDS = 1


class GestureRecognitionEngine:
    """
    Owns one loaded model and one HandLandmarkDetector. Construction is
    the "expensive" part (loading a model file, spinning up MediaPipe) -
    create ONE instance and reuse it across requests (see get_engine()
    below) rather than constructing a new engine per prediction.
    """

    def __init__(
        self,
        model_version: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        detector: Optional[HandLandmarkDetector] = None,
    ):
        self.model, self.model_metadata = load_model(version=model_version)
        self.model_version = self.model_metadata["version"]
        self.confidence_threshold = confidence_threshold
        # static_image_mode=True: each predict() call is an independent
        # image (an API upload), not consecutive frames of one video
        # stream where MediaPipe could track the hand across frames.
        #
        # max_num_hands=2 (not 1): this is deliberately higher than the
        # number of hands the model actually supports (MAX_SUPPORTED_HANDS
        # = 1). If it were set to 1, MediaPipe would silently return only
        # its single most-confident hand even when a second hand is
        # present in the frame, making a genuine two-hand frame
        # indistinguishable from a clean one-hand frame. Detecting up to
        # 2 lets predict() below actually notice the multi-hand case and
        # reject it explicitly instead of silently guessing.
        self.detector = detector or HandLandmarkDetector(static_image_mode=True, max_num_hands=4)

        logger.info(
            "GestureRecognitionEngine ready (model_version=%s, classes=%d, confidence_threshold=%.2f)",
            self.model_version, self.model_metadata["num_classes"], self.confidence_threshold,
        )

    def predict(self, image) -> PredictionResult:
        """
        Runs the full pipeline on one image and always returns a
        PredictionResult - success or a graceful, informative failure.
        `image` is a BGR frame as OpenCV represents it (e.g. the result
        of cv2.imread(...) or cv2.imdecode(...) on an uploaded file).
        """
        start = time.perf_counter()

        try:
            detection = self.detector.extract_landmarks_with_metadata(image)
            hand_count = detection["hand_count"]
            has_person = detection.get("has_person", False)
            upper_body_visible = detection.get("upper_body_visible", False)
            partial_hand_visible = detection.get("partial_hand_visible", False)
            hand_centered = detection.get("hand_centered", False)

            if hand_count == 0:
                return self._failure(
                    "No hand detected in image.",
                    start,
                    has_person=has_person,
                    hand_count=hand_count,
                    upper_body_visible=upper_body_visible,
                    partial_hand_visible=partial_hand_visible,
                    hand_centered=hand_centered,
                )

            if hand_count > MAX_SUPPORTED_HANDS:
                return self._failure(
                    f"Detected {hand_count} hands, but this model only supports "
                    f"{MAX_SUPPORTED_HANDS} at a time. Please show one hand only.",
                    start,
                    has_person=has_person,
                    hand_count=hand_count,
                    upper_body_visible=upper_body_visible,
                    partial_hand_visible=partial_hand_visible,
                    hand_centered=hand_centered,
                )

            landmarks = detection["landmarks"]

            try:
                feature_vector = build_feature_vector(landmarks)
            except FeatureValidationError as e:
                return self._failure(
                    f"Feature validation failed: {e}",
                    start,
                    has_person=has_person,
                    hand_count=hand_count,
                    upper_body_visible=upper_body_visible,
                    partial_hand_visible=partial_hand_visible,
                    hand_centered=hand_centered,
                )

            model_start = time.perf_counter()
            probabilities_row = self.model.predict_proba(feature_vector)[0]
            model_time_ms = (time.perf_counter() - model_start) * 1000

            probabilities = {
                str(class_label): float(prob)
                for class_label, prob in zip(self.model.classes_, probabilities_row)
            }
            predicted_class = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_class]
            above_threshold = confidence >= self.confidence_threshold

            total_time_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "prediction: class=%s confidence=%.4f above_threshold=%s "
                "model_version=%s model_time_ms=%.2f total_time_ms=%.2f",
                predicted_class, confidence, above_threshold,
                self.model_version, model_time_ms, total_time_ms,
            )

            return PredictionResult.ok(
                predicted_class=predicted_class,
                confidence=confidence,
                above_confidence_threshold=above_threshold,
                probabilities=probabilities,
                model_version=self.model_version,
                model_inference_time_ms=round(model_time_ms, 4),
                total_time_ms=round(total_time_ms, 4),
                landmarks=landmarks,
                has_person=has_person,
                hand_count=hand_count,
                upper_body_visible=upper_body_visible,
                partial_hand_visible=partial_hand_visible,
                hand_centered=hand_centered,
            )

        except Exception as e:
            # Deliberately broad: predict() must never raise. A corrupt
            # image, an unexpected OpenCV/MediaPipe error, etc. should
            # degrade to a failed PredictionResult, not crash the
            # caller's request handler.
            logger.exception("Unexpected error during prediction")
            return self._failure(f"Unexpected error during prediction: {e}", start)

    def _failure(
        self,
        message: str,
        start_time: float,
        has_person: bool = False,
        hand_count: int = 0,
        upper_body_visible: bool = False,
        partial_hand_visible: bool = False,
        hand_centered: bool = False,
    ) -> PredictionResult:
        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("prediction failed: %s (total_time_ms=%.2f)", message, total_time_ms)
        return PredictionResult.failure(
            error=message,
            model_version=self.model_version,
            total_time_ms=round(total_time_ms, 4),
            has_person=has_person,
            hand_count=hand_count,
            upper_body_visible=upper_body_visible,
            partial_hand_visible=partial_hand_visible,
            hand_centered=hand_centered,
        )

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# --- Module-level singleton, so the app doesn't reload the model / ---
# --- spin up a new MediaPipe instance on every request.            ---

_engine: Optional[GestureRecognitionEngine] = None


def get_engine() -> GestureRecognitionEngine:
    """
    Returns the shared GestureRecognitionEngine instance, creating it on
    first use. Raises ModelNotFoundError/ModelVersionError immediately
    (rather than lazily failing per-request) if no model has been
    exported yet - see app/ai/ml/training/export_model.py.
    """
    global _engine
    if _engine is None:
        _engine = GestureRecognitionEngine()
    return _engine


def predict(image) -> PredictionResult:
    """
    The public interface of this entire AI module:

        from app.ai.ml.inference.engine import predict
        result = predict(image)

    Everything else in app/ai/ is an implementation detail behind this
    one function. `image` is a decoded BGR frame (e.g. via cv2.imread,
    or app.ai.handtracking.detector.decode_image for raw upload bytes).
    """
    return get_engine().predict(image)


def predict_from_bytes(image_bytes: bytes) -> PredictionResult:
    """
    Same as predict(), but takes raw image bytes directly - e.g. the
    contents of a FastAPI UploadFile - instead of an already-decoded
    frame. This is the entry point routers/services should use, so
    that nothing outside app/ai/handtracking/detector.py ever needs to
    import cv2 just to decode an uploaded image.
    """
    start = time.perf_counter()
    image = decode_image(image_bytes)
    if image is None:
        return get_engine()._failure(
            "Could not decode image data - the uploaded file may be corrupt "
            "or not a supported image format.",
            start_time=start,
        )
    return predict(image)