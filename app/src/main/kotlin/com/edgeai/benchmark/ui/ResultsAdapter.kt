package com.edgeai.benchmark.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.edgeai.benchmark.R
import com.edgeai.benchmark.model.BenchmarkResult

class ResultsAdapter : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {

    private val items = mutableListOf<BenchmarkResult>()

    fun addResult(result: BenchmarkResult) {
        items.add(0, result)
        notifyItemInserted(0)
    }

    fun clear() {
        val size = items.size
        items.clear()
        notifyItemRangeRemoved(0, size)
    }

    override fun getItemCount() = items.size

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_result, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(items[position])
    }

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        private val tvModelName: TextView = view.findViewById(R.id.tvModelName)
        private val tvConfig: TextView = view.findViewById(R.id.tvConfig)
        private val tvP50: TextView = view.findViewById(R.id.tvP50)
        private val tvP95: TextView = view.findViewById(R.id.tvP95)
        private val tvP99: TextView = view.findViewById(R.id.tvP99)
        private val tvMemory: TextView = view.findViewById(R.id.tvMemory)
        private val tvColdStart: TextView = view.findViewById(R.id.tvColdStart)
        private val tvThermal: TextView = view.findViewById(R.id.tvThermal)

        fun bind(r: BenchmarkResult) {
            tvModelName.text = r.modelName
            tvConfig.text = "${r.runtime.label} · ${r.backend.label} · ${r.precision.label}"
            tvP50.text = "%.1f".format(r.p50LatencyMs)
            tvP95.text = "%.1f".format(r.p95LatencyMs)
            tvP99.text = "%.1f".format(r.p99LatencyMs)
            tvMemory.text = "%.1f".format(r.peakMemoryMb)
            tvColdStart.text = "Cold start: %.1f ms".format(r.coldStartMs)
            tvThermal.text = "${r.thermalBefore.name} → ${r.thermalAfter.name}"
        }
    }
}
