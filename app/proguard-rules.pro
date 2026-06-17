# TFLite / LiteRT
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }

# ONNX Runtime (v0.2)
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# ExecuTorch (v0.3)
-keep class org.pytorch.executorch.** { *; }
-keep class com.facebook.jni.** { *; }
-keep class com.facebook.soloader.** { *; }
-dontwarn javax.annotation.**
-dontwarn com.facebook.jni.**

# Optional classes referenced by TensorFlow Lite support/gpu artifacts.
-dontwarn com.google.auto.value.AutoValue$Builder
-dontwarn com.google.auto.value.AutoValue
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options$GpuBackend
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options
