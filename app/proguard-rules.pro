# TFLite / LiteRT
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }

# ONNX Runtime (v0.2)
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Optional classes referenced by TensorFlow Lite support/gpu artifacts.
-dontwarn com.google.auto.value.AutoValue$Builder
-dontwarn com.google.auto.value.AutoValue
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options$GpuBackend
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options
