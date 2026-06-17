package com.edgeai.benchmark.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
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
import com.edgeai.benchmark.benchmark.LiteRtEngine
import com.edgeai.benchmark.benchmark.OnnxEngine
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.BenchmarkResult
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.util.CsvExporter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var spinnerRuntime: Spinner
    private lateinit var spinnerModel: Spinner
    private lateinit var spinnerBackend: Spinner
    private lateinit var spinnerPrecision: Spinner
    private lateinit var btnRun: Button
    private lateinit var tvStatus: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var recyclerResults: RecyclerView
    private lateinit var adapter: ResultsAdapter

    // Models dir: adb push ../models/ /sdcard/Android/data/com.edgeai.benchmark/files/models/
    private val modelsDir: File get() = File(getExternalFilesDir(null), "models")

    // Mapping from spinner index to filename prefix
    private val modelFileNames = listOf(
        "mobilenet_v3_small",
        "efficientnet_lite0"
    )

    // File extension per runtime
    private val runtimeExtensions = listOf("tflite", "onnx")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupRecyclerView()
        requestStoragePermission()

        btnRun.setOnClickListener { startBenchmark() }
    }

    private fun bindViews() {
        spinnerRuntime = findViewById(R.id.spinnerRuntime)
        spinnerModel = findViewById(R.id.spinnerModel)
        spinnerBackend = findViewById(R.id.spinnerBackend)
        spinnerPrecision = findViewById(R.id.spinnerPrecision)
        btnRun = findViewById(R.id.btnRun)
        tvStatus = findViewById(R.id.tvStatus)
        progressBar = findViewById(R.id.progressBar)
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

        val precision = when (spinnerPrecision.selectedItemPosition) {
            1 -> Precision.INT8
            else -> Precision.FP32
        }

        val backend = when (spinnerBackend.selectedItemPosition) {
            1 -> Backend.GPU_DELEGATE
            else -> Backend.CPU
        }

        val precisionSuffix = if (precision == Precision.INT8) "int8" else "fp32"
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
                        1 -> OnnxEngine(context = this@MainActivity, backend = backend)
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
                tvStatus.text = getString(R.string.status_done)
            }.onFailure { error ->
                tvStatus.text = getString(R.string.status_error, error.message)
            }

            setRunning(false)
        }
    }

    private fun saveToCsv(result: BenchmarkResult) {
        runCatching {
            val file = CsvExporter.append(this, result)
            runOnUiThread {
                Toast.makeText(this, "Saved: ${file.name}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setRunning(running: Boolean) {
        btnRun.isEnabled = !running
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
