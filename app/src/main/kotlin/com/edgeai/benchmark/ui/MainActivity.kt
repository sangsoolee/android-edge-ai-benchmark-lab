package com.edgeai.benchmark.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.edgeai.benchmark.R
import com.edgeai.benchmark.benchmark.BenchmarkEngine
import com.edgeai.benchmark.benchmark.ExecuTorchEngine
import com.edgeai.benchmark.benchmark.LiteRtEngine
import com.edgeai.benchmark.benchmark.OnnxEngine
import com.edgeai.benchmark.detection.Detection
import com.edgeai.benchmark.detection.DetectionBenchmarkResult
import com.edgeai.benchmark.detection.DetectionCsvExporter
import com.edgeai.benchmark.detection.DetectionJsonWriter
import com.edgeai.benchmark.detection.DetectionRenderer
import com.edgeai.benchmark.detection.Percentiles
import com.edgeai.benchmark.detection.YoloDetector
import com.edgeai.benchmark.detection.YoloOutputDecoder
import com.edgeai.benchmark.detection.Nms
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.BenchmarkResult
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import com.edgeai.benchmark.util.CsvExporter
import com.edgeai.benchmark.util.MemoryTracker
import com.edgeai.benchmark.util.ThermalMonitor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var spinnerRuntime: Spinner
    private lateinit var spinnerModel: Spinner
    private lateinit var spinnerBackend: Spinner
    private lateinit var spinnerPrecision: Spinner
    private lateinit var btnRun: Button
    private lateinit var btnDetect: Button
    private lateinit var btnDetectBench: Button
    private lateinit var tvStatus: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var ivDetection: ImageView
    private lateinit var recyclerResults: RecyclerView
    private lateinit var adapter: ResultsAdapter

    // Models dir: adb push ../models/ /sdcard/Android/data/com.edgeai.benchmark/files/models/
    private val modelsDir: File get() = File(getExternalFilesDir(null), "models")
    // Sample image for detection: adb push img.jpg .../files/samples/sample.jpg
    private val sampleImage: File get() = File(getExternalFilesDir(null), "samples/sample.jpg")
    private val detectionOutDir: File get() = File(getExternalFilesDir(null), "results/detection")

    // Mapping from spinner index to filename prefix.
    // Must match SUPPORTED_MODELS in scripts/convert/export_*.py.
    private val modelFileNames = listOf(
        "mobilenet_v3_small",
        "efficientnet_b0",
        // v0.5: YOLOv8n detection. LiteRtEngine reads I/O sizes from the model, so
        // inference latency works without code changes. NOTE: ONNX/ExecuTorch engines
        // still hard-code a 224x224 input — use LiteRT for YOLOv8n until v0.5.x.
        "yolov8n"
    )

    // File extension per runtime
    private val runtimeExtensions = listOf("tflite", "onnx", "pte")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupRecyclerView()
        setupBackendSpinnerLink()
        requestStoragePermission()

        btnRun.setOnClickListener { startBenchmark() }
        btnDetect.setOnClickListener { startDetection() }
        btnDetectBench.setOnClickListener { startDetectionBenchmark() }
    }

    /**
     * Backend options are runtime-specific so the user can't pick a backend the
     * engine doesn't actually run (which would mislabel the CSV):
     *   LiteRT     → CPU, GPU delegate
     *   ONNX       → CPU, NNAPI
     *   ExecuTorch → CPU
     */
    private fun setupBackendSpinnerLink() {
        spinnerRuntime.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val arrayRes = when (position) {
                    0 -> R.array.backend_options            // LiteRT: CPU / GPU
                    1 -> R.array.backend_options_onnx       // ONNX: CPU / NNAPI
                    else -> R.array.backend_options_cpu_only // ExecuTorch: CPU
                }
                spinnerBackend.adapter = ArrayAdapter.createFromResource(
                    this@MainActivity, arrayRes, android.R.layout.simple_spinner_item
                ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun bindViews() {
        spinnerRuntime = findViewById(R.id.spinnerRuntime)
        spinnerModel = findViewById(R.id.spinnerModel)
        spinnerBackend = findViewById(R.id.spinnerBackend)
        spinnerPrecision = findViewById(R.id.spinnerPrecision)
        btnRun = findViewById(R.id.btnRun)
        btnDetect = findViewById(R.id.btnDetect)
        btnDetectBench = findViewById(R.id.btnDetectBench)
        tvStatus = findViewById(R.id.tvStatus)
        progressBar = findViewById(R.id.progressBar)
        ivDetection = findViewById(R.id.ivDetection)
        recyclerResults = findViewById(R.id.recyclerResults)
    }

    private fun setupRecyclerView() {
        adapter = ResultsAdapter()
        recyclerResults.layoutManager = LinearLayoutManager(this)
        recyclerResults.adapter = adapter
    }

    private fun startBenchmark() {
        val runtimeIndex = spinnerRuntime.selectedItemPosition
        val modelIndex = spinnerModel.selectedItemPosition
        val modelFileName = modelFileNames[modelIndex]
        val modelName = spinnerModel.selectedItem.toString()

        // Index order must match precision_options: FP32(0), FP16(1), INT8(2).
        val precision = when (spinnerPrecision.selectedItemPosition) {
            1 -> Precision.FP16
            2 -> Precision.INT8
            else -> Precision.FP32
        }

        // Backend meaning depends on the selected runtime (see setupBackendSpinnerLink).
        val backendIndex = spinnerBackend.selectedItemPosition
        val backend = when (runtimeIndex) {
            0 -> if (backendIndex == 1) Backend.GPU_DELEGATE else Backend.CPU  // LiteRT
            1 -> if (backendIndex == 1) Backend.NNAPI else Backend.CPU         // ONNX
            else -> Backend.CPU                                                // ExecuTorch
        }

        // fp32 / fp16 / int8 — matches export_tflite.py output naming
        val precisionSuffix = precision.label.lowercase()
        val ext = runtimeExtensions[runtimeIndex]
        val modelPath = File(modelsDir, "${modelFileName}_${precisionSuffix}.$ext").absolutePath

        if (!File(modelPath).exists()) {
            tvStatus.text = getString(R.string.status_model_missing)
            Toast.makeText(this, "Model not found:\n$modelPath", Toast.LENGTH_LONG).show()
            return
        }

        setRunning(true)

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val engine: BenchmarkEngine = when (runtimeIndex) {
                        1 -> OnnxEngine(context = this@MainActivity, requestedBackend = backend)
                        2 -> ExecuTorchEngine(context = this@MainActivity, requestedBackend = backend)
                        else -> LiteRtEngine(context = this@MainActivity, backend = backend)
                    }
                    engine.benchmark(
                        modelPath = modelPath,
                        modelName = modelName,
                        precision = precision
                    )
                }
            }

            result.onSuccess { benchmarkResult ->
                adapter.addResult(benchmarkResult)
                saveToCsv(benchmarkResult)
            }.onFailure { error ->
                tvStatus.text = getString(R.string.status_error, error.message)
            }

            setRunning(false)
        }
    }

    private fun saveToCsv(result: BenchmarkResult) {
        try {
            val file = CsvExporter.append(this, result)
            tvStatus.text = getString(R.string.status_done)
            Toast.makeText(this, "Saved: ${file.name}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("MainActivity", "CSV save failed for ${result.runtime.label}", e)
            tvStatus.text = "Done. CSV save failed: ${e.javaClass.simpleName}: ${e.message}"
            Toast.makeText(this, "CSV save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // ---------------------------------------------------------------------------
    // Detection (v0.5.2): run YOLOv8n FP32 on a pushed sample image, render + save.
    // ---------------------------------------------------------------------------

    private fun startDetection() {
        val modelPath = File(modelsDir, "yolov8n_fp32.tflite").absolutePath
        if (!File(modelPath).exists()) {
            tvStatus.text = getString(R.string.status_model_missing)
            Toast.makeText(this, "Model not found:\n$modelPath", Toast.LENGTH_LONG).show()
            return
        }
        if (!sampleImage.exists()) {
            tvStatus.text = getString(R.string.status_no_sample)
            return
        }

        setRunning(true)
        tvStatus.text = getString(R.string.status_detecting)

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val src = BitmapFactory.decodeFile(sampleImage.absolutePath)
                        ?: error("Could not decode ${sampleImage.name}")
                    val dets: List<Detection> = YoloDetector(modelPath).use { it.detect(src) }
                    val annotated = DetectionRenderer.render(src, dets)
                    detectionOutDir.mkdirs()
                    savePng(annotated, File(detectionOutDir, "android_sample.png"))
                    DetectionJsonWriter.write(dets, File(detectionOutDir, "android_sample.json"))
                    annotated to dets
                }
            }

            result.onSuccess { (bmp, dets) ->
                ivDetection.setImageBitmap(bmp)
                ivDetection.visibility = View.VISIBLE
                val top = dets.take(3).joinToString(", ") { "${it.label} ${"%.2f".format(it.score)}" }
                tvStatus.text = "Detected ${dets.size}: $top"
                Toast.makeText(this@MainActivity, "Saved android_sample.json / .png", Toast.LENGTH_SHORT).show()
            }.onFailure { e ->
                Log.e("MainActivity", "Detection failed", e)
                tvStatus.text = getString(R.string.status_error, e.message)
            }

            setRunning(false)
        }
    }

    private fun savePng(bmp: Bitmap, file: File) {
        FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    // ---------------------------------------------------------------------------
    // Detection 3-phase benchmark (v0.5.3): YOLOv8n LiteRT, FP32 + INT8.
    // Phases: preprocess / inference / postprocess + end-to-end (own outer timer).
    // No rendering in the loop. Writes a dedicated detection CSV.
    // ---------------------------------------------------------------------------

    private fun startDetectionBenchmark(warmup: Int = 20, measured: Int = 100) {
        if (!sampleImage.exists()) {
            tvStatus.text = getString(R.string.status_no_sample)
            return
        }
        setRunning(true)
        tvStatus.text = getString(R.string.status_detect_bench)

        lifecycleScope.launch {
            val results = withContext(Dispatchers.Default) {
                runCatching {
                    val src = BitmapFactory.decodeFile(sampleImage.absolutePath)
                        ?: error("Could not decode ${sampleImage.name}")
                    val out = ArrayList<DetectionBenchmarkResult>()
                    for ((precision, suffix) in listOf(Precision.FP32 to "fp32", Precision.INT8 to "int8")) {
                        val modelFile = File(modelsDir, "yolov8n_$suffix.tflite")
                        if (modelFile.exists()) {
                            out.add(benchmarkDetection(src, modelFile, precision, warmup, measured))
                        }
                    }
                    out
                }
            }

            results.onSuccess { list ->
                if (list.isEmpty()) {
                    tvStatus.text = "No yolov8n_*.tflite models found in ${modelsDir.name}/"
                } else {
                    val file = list.fold<DetectionBenchmarkResult, File?>(null) { _, r ->
                        DetectionCsvExporter.append(this@MainActivity, r)
                    }
                    val summary = list.joinToString("  |  ") {
                        "${it.precision.label}: e2e ${"%.1f".format(it.endToEndP50)}ms " +
                        "(pre ${"%.1f".format(it.preprocessP50)} / inf ${"%.1f".format(it.inferenceP50)} / post ${"%.1f".format(it.postprocessP50)})"
                    }
                    tvStatus.text = summary
                    Toast.makeText(this@MainActivity, "Saved ${file?.name}", Toast.LENGTH_SHORT).show()
                }
            }.onFailure { e ->
                Log.e("MainActivity", "Detection benchmark failed", e)
                tvStatus.text = getString(R.string.status_error, e.message)
            }

            setRunning(false)
        }
    }

    private fun benchmarkDetection(
        src: Bitmap, modelFile: File, precision: Precision, warmup: Int, measured: Int
    ): DetectionBenchmarkResult {
        val thermalBefore = ThermalMonitor.currentStatus(this)
        val detector = YoloDetector(modelFile.absolutePath)
        try {
            repeat(warmup) { detector.detectTimed(src) }

            val memBefore = MemoryTracker.sampleNowMb()
            val pre = DoubleArray(measured); val inf = DoubleArray(measured)
            val post = DoubleArray(measured); val e2e = DoubleArray(measured)
            val counts = IntArray(measured)

            val tracker = MemoryTracker(this)
            tracker.start()
            for (i in 0 until measured) {
                val t = detector.detectTimed(src)
                pre[i] = t.preprocessMs; inf[i] = t.inferenceMs
                post[i] = t.postprocessMs; e2e[i] = t.endToEndMs; counts[i] = t.detectionCount
            }
            val memPeak = tracker.stopAndGetPeakMb()
            val memAfter = MemoryTracker.sampleNowMb()
            val thermalAfter = ThermalMonitor.currentStatus(this)

            val sortedCounts = counts.sortedArray()
            val countP50 = sortedCounts[((sortedCounts.size - 1) / 2)]

            return DetectionBenchmarkResult(
                timestampUtcMs = System.currentTimeMillis(),
                modelName = "YOLOv8n", runtime = Runtime.LITERT, backend = Backend.CPU,
                precision = precision, inputWidth = 640, inputHeight = 640,
                warmupRuns = warmup, measuredRuns = measured,
                preprocessP50 = Percentiles.of(pre, 50.0), preprocessP90 = Percentiles.of(pre, 90.0), preprocessP99 = Percentiles.of(pre, 99.0),
                inferenceP50 = Percentiles.of(inf, 50.0), inferenceP90 = Percentiles.of(inf, 90.0), inferenceP99 = Percentiles.of(inf, 99.0),
                postprocessP50 = Percentiles.of(post, 50.0), postprocessP90 = Percentiles.of(post, 90.0), postprocessP99 = Percentiles.of(post, 99.0),
                endToEndP50 = Percentiles.of(e2e, 50.0), endToEndP90 = Percentiles.of(e2e, 90.0), endToEndP99 = Percentiles.of(e2e, 99.0),
                detectionCountP50 = countP50,
                confThreshold = YoloOutputDecoder.CONF_THRES, iouThreshold = Nms.IOU_THRES, maxDetections = Nms.MAX_DET,
                modelSizeMb = modelFile.length() / (1024.0 * 1024.0),
                memBeforeMb = memBefore, memPeakMb = memPeak, memAfterMb = memAfter,
                thermalBefore = thermalBefore, thermalAfter = thermalAfter,
                deviceModel = Build.MODEL, deviceChip = Build.HARDWARE,
                androidVersion = Build.VERSION.SDK_INT,
                abiName = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
            )
        } finally {
            detector.close()
        }
    }

    private fun setRunning(running: Boolean) {
        btnRun.isEnabled = !running
        btnDetect.isEnabled = !running
        btnDetectBench.isEnabled = !running
        progressBar.visibility = if (running) View.VISIBLE else View.GONE
        if (running) tvStatus.text = getString(R.string.status_running)
    }

    // ---------------------------------------------------------------------------
    // Storage permission (needed to read model files on older APIs)
    // ---------------------------------------------------------------------------

    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) return // scoped storage, no-op
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) return        // MANAGE_EXTERNAL_STORAGE needs Settings intent, skip for now

        val perm = Manifest.permission.READ_EXTERNAL_STORAGE
        if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(perm), REQUEST_STORAGE)
        }
    }

    companion object {
        private const val REQUEST_STORAGE = 1001
    }
}
