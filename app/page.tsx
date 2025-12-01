'use client';
import { useState, useEffect } from 'react';
import { client } from "@gradio/client";
import { Upload, Activity, AlertCircle, CheckCircle2, ChevronRight, ShieldAlert, HeartPulse, Brain, Download, Database } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import NiftiViewer to avoid SSR issues
const NiftiViewer = dynamic(() => import('./components/NiftiViewer'), { ssr: false });

// Use the local Next.js proxy rewrite path
// DEFINED ONCE AT COMPONENT SCOPE
// const HF_SPACE_URL = "https://ai4all3dcnn-neuroscan-backend.hf.space";
const HF_SPACE_URL = "https://cooleschimo-neuroscan-backend.hf.space";
// const HF_SPACE_URL = "http://127.0.0.1:7860";  // Local backend (only for local dev)

interface AnalysisResult {
  predictions: [string, number][];
  scan_data: string | null;
  heatmap_data: string | null;
  predicted_class: string;
  confidence: number;
}

export default function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    // Handle scroll for navbar styling
    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 50);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        setResult(null);

        try {
            const app = await client(HF_SPACE_URL);
            console.log("Gradio client initialized.");

            const response = await app.predict("/predict", [file]) as any;

            // DEBUG: Log the entire response to see what we're getting
            console.log("Full response:", JSON.stringify(response, null, 2));
            console.log("response.data:", response.data);
            console.log("response.data[0]:", response.data?.[0]);

            // The new backend returns a JSON object
            if (!response?.data?.[0]) {
                throw new Error("Invalid response format from model");
            }

            const analysisResult: AnalysisResult = response.data[0];

            setResult(analysisResult);
            console.log("Prediction successful:", analysisResult);

        } catch (error: any) {
            console.error("Error during NeuroScan prediction:", error);

            let errorMessage = "Something went wrong connecting to the model. Check the console for details.";

            if (error instanceof Error && error.message) {
                if (error.message.includes("fetch") || error.message.includes("Failed to fetch")) {
                    errorMessage = "Connection Failed. Check if the Gradio space is awake/running.";
                } else if (error.message.includes("404")) {
                    errorMessage = "Model endpoint not found.";
                } else if (error.message.includes("Invalid response format")) {
                    errorMessage = "Invalid response format from model. Check console for details.";
                } else {
                    errorMessage = `Model Error: ${error.message}`;
                }
            }

            alert(errorMessage);

        } finally {
            setLoading(false);
        }
    };

    const scrollTo = (id: string) => {
        const element = document.getElementById(id);
        if (element) {
            const y = element.getBoundingClientRect().top + window.scrollY - 100;
            window.scrollTo({top: y, behavior: 'smooth'});
        }
    };

    // ... (rest of the return JSX) ...
    return (
        <main className="min-h-screen font-sans selection:bg-cyan-100">
          
          {/* ================= BUBBLE NAVBAR ================= */}
          <div className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4 pointer-events-none">
            <nav className={`pointer-events-auto glass-panel rounded-full px-6 py-3 flex items-center gap-8 transition-all duration-300 ${scrolled ? 'scale-95 shadow-xl bg-white/90' : 'scale-100 bg-white/70'}`}>
              {/* Logo */}
              <div className="flex items-center gap-2 cursor-pointer" onClick={() => window.scrollTo(0,0)}>
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-600 to-blue-700 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                  <Activity className="w-4 h-4 text-white" />
                </div>
                <span className="font-bold text-lg tracking-tight text-slate-800 hidden sm:block">
                  NeuroScan
                </span>
              </div>
    
              {/* Links */}
              <div className="hidden md:flex items-center gap-2 text-sm font-semibold text-slate-500">
                <button onClick={() => scrollTo('impact')} className="hover:text-cyan-700 hover:bg-slate-100 px-4 py-2 rounded-full transition-all">Impact</button>
                <button onClick={() => scrollTo('data-section')} className="hover:text-cyan-700 hover:bg-slate-100 px-4 py-2 rounded-full transition-all flex items-center gap-2">
                    Data & Download
                </button>
              </div>
    
              {/* CTA Button */}
              <button 
                onClick={() => scrollTo('diagnosis')}
                className="bg-slate-900 text-white hover:bg-slate-800 px-5 py-2 rounded-full text-sm font-bold transition-colors flex items-center gap-2 shadow-lg shadow-slate-900/10"
              >
                Run Demo <ChevronRight className="w-3 h-3" />
              </button>
            </nav>
          </div>
    
          {/* ================= HERO SECTION ================= */}
          <section className="relative pt-48 pb-32 px-6 overflow-hidden bg-slate-50">
            {/* Background Gradients */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-cyan-200/40 rounded-[100%] blur-[100px] -z-10 opacity-60"></div>
            <div className="absolute top-20 right-0 w-[400px] h-[400px] bg-blue-200/30 rounded-[100%] blur-[80px] -z-10 opacity-50"></div>
            
            <div className="max-w-4xl mx-auto text-center space-y-8">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-200 bg-white shadow-sm text-cyan-700 text-xs font-bold uppercase tracking-wider">
                <span className="w-2 h-2 rounded-full bg-cyan-500 glow-point"></span>
                V5 Model Online
              </div>
              
              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.1] text-slate-900">
                Precision Diagnostics for<br />
                <span className="text-gradient">Intracranial Vascular Health</span>
              </h1>
    
              <p className="text-lg md:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed font-medium">
                Bridging the gap in radiological review with deep learning. 
                Detecting the critical 3mm anomalies that human vision misses.
              </p>
    
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                <button onClick={() => scrollTo('diagnosis')} className="w-full sm:w-auto px-8 py-4 bg-cyan-600 hover:bg-cyan-500 text-white rounded-full font-bold transition-all shadow-xl shadow-cyan-500/20 hover:scale-105">
                  Launch Diagnostic Tool
                </button>
                <button onClick={() => scrollTo('impact')} className="w-full sm:w-auto px-8 py-4 bg-white text-slate-600 border border-slate-200 hover:border-cyan-300 hover:text-cyan-700 rounded-full font-bold transition-all shadow-sm">
                  Why This Matters
                </button>
              </div>
            </div>
          </section>
    
          {/* ================= IMPORTANCE / IMPACT SECTION ================= */}
          <section id="impact" className="py-24 px-6 bg-white">
            <div className="max-w-6xl mx-auto">
              <div className="grid md:grid-cols-2 gap-20 items-center">
                
                <div className="space-y-10">
                  <h2 className="text-4xl font-extrabold text-slate-900">The Silent Threat</h2>
                  <div className="space-y-8">
                    <div className="flex gap-5">
                      <div className="mt-1 bg-rose-50 p-3 rounded-2xl h-fit border border-rose-100">
                        <ShieldAlert className="w-6 h-6 text-rose-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">Critical Mortality Rates</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          Brain aneurysms often show no symptoms until rupture. Once ruptured, the mortality rate is approximately <span className="text-rose-600 font-black">50%</span>.
                        </p>
                      </div>
                    </div>
    
                    <div className="flex gap-5">
                      <div className="mt-1 bg-amber-50 p-3 rounded-2xl h-fit border border-amber-100">
                        <AlertCircle className="w-6 h-6 text-amber-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">The 3mm Blind Spot</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          Aneurysms under 3mm are notoriously difficult to spot in complex 3D MRA scans, leading to potentially missed early interventions.
                        </p>
                      </div>
                    </div>
    
                      <div className="flex gap-5">
                      <div className="mt-1 bg-cyan-50 p-3 rounded-2xl h-fit border border-cyan-100">
                        <HeartPulse className="w-6 h-6 text-cyan-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">The AI Advantage</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          NeuroScan acts as a "second pair of eyes" to support clinical teams. While human radiologists generally maintain superior overall diagnostic accuracy, NeuroScan excels in detecting small, high-risk anomalies that are often missed due to human fatigue or visual limitations.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
    
                {/* Visual Stats (Gradient Background) */}
                <div className="relative p-4 pt-10">
                  {/* The subtle gradient effect (made larger and more central) */}
                  <div className="absolute inset-0 transform scale-150 bg-gradient-to-tr from-cyan-300/30 via-rose-300/30 to-blue-300/30 rounded-full blur-[150px] -z-10 opacity-80 animate-pulse-slow"></div>
    
                  <div className="bg-white p-4 space-y-12 rounded-xl">
                    <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-slate-900 tracking-tight">6.5M</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">People in the US with unruptured aneurysms</div>
                    </div>
                    
                    <div className="w-16 h-1 bg-slate-200 mx-auto rounded-full"></div>
    
                      <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-rose-600 tracking-tight">30k</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">Ruptures occur each year</div>
                    </div>
                    
                    <div className="w-16 h-1 bg-slate-200 mx-auto rounded-full"></div>
    
                      <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-cyan-700 tracking-tight">~15%</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">Die before reaching the hospital</div>
                    </div>
                  </div>
    
                </div>
                
                {/* Sources Div, spanning both columns (or just placed after the grid) */}
              </div>
              <div className="pt-16 text-center">
                  <p className="text-sm text-slate-500 max-w-3xl mx-auto">
                    *Sources: 6.5 million prevalence and 30,000 annual ruptures based on common NIH/CDC estimates. 15% pre-hospital mortality cited in clinical reports, including data from the Barrow Neurological Institute (2023).
                  </p>
              </div>
            </div>
          </section>
    
          {/* ================= SCIENCE & DATA SECTION ================= */}
          <section id="data-section" className="py-24 px-6 bg-slate-50 border-t border-slate-200">
            <div className="max-w-6xl mx-auto grid lg:grid-cols-12 gap-12">
              
              <div className="lg:col-span-7 space-y-12">
                <div>
                  <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-cyan-100 rounded-lg text-cyan-700"><Database className="w-6 h-6" /></div>
                      <h2 className="text-3xl font-bold text-slate-900">Benchmarks & Data</h2>
                  </div>
                  <p className="text-slate-600 text-lg leading-relaxed">
                    We benchmarked our V5 Model against historical radiologist performance data. The results show a massive improvement in "recall" (sensitivity) for small, difficult-to-detect vessels.
                  </p>
                </div>
    
                {/* Chart Image - White bg, subtle shadow */}
                <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-lg">
                                    {/* <img 
                    src="/benchmark-chart.png" 
                    alt="Benchmark Chart" 
                    className="w-full h-auto rounded-lg"
                  /> */}
                </div>
                
                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm">
                    <div className="text-sm text-slate-500 font-bold mb-1">Human Sensitivity (&lt;3mm)</div>
                    <div className="text-4xl font-black text-slate-400">38.0%</div>
                  </div>
                  <div className="p-6 rounded-2xl bg-cyan-600 text-white shadow-xl shadow-cyan-600/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-2 opacity-20"><Brain className="w-12 h-12 text-white" /></div>
                    <div className="text-sm text-cyan-100 font-bold mb-1">NeuroScan V5 Sensitivity</div>
                    <div className="text-4xl font-black text-white">83.7%</div>
                  </div>
                </div>
              </div>
    
              {/* Download Box */}
              <div className="lg:col-span-5 flex flex-col justify-center">
                <div className="bg-white p-10 rounded-3xl text-center space-y-8 border border-slate-200 shadow-2xl">
                  <div className="w-20 h-20 mx-auto bg-cyan-50 rounded-2xl flex items-center justify-center text-cyan-600 mb-4 border border-cyan-100">
                    <Download className="w-10 h-10" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-slate-900 mb-2">Validate the Findings</h3>
                    <p className="text-slate-500 text-base">
                      We believe in transparency. Download our anonymized test sample (NIfTI format) to run your own diagnostic check using the tool below.
                    </p>
                  </div>
                  <a 
                    href="/sample_scan.nii" 
                    download
                    className="block w-full py-4 bg-slate-900 text-white hover:bg-slate-800 font-bold text-lg rounded-xl transition-colors shadow-lg shadow-slate-900/10"
                  >
                    Download Test Sample
                  </a>
                </div>
              </div>
            </div>
          </section>
    
          {/* ================= APP INTERFACE ================= */}
          <section id="diagnosis" className="py-24 px-6 relative bg-white">
            <div className="max-w-4xl mx-auto">
              
              <div className="bg-white rounded-3xl overflow-hidden border border-slate-200 shadow-2xl">
                {/* Interface Header */}
                <div className="bg-slate-50 p-8 border-b border-slate-200 flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold flex items-center gap-2 text-slate-900">
                      <Activity className="w-6 h-6 text-cyan-600" />
                      Diagnostic Interface
                    </h2>
                    <p className="text-slate-500 text-sm mt-1">Ready for 3D Volumetric Analysis</p>
                  </div>
                  <div className="flex gap-2">
                      <div className="w-3 h-3 rounded-full bg-rose-400/30 border border-rose-400"></div>
                      <div className="w-3 h-3 rounded-full bg-amber-400/30 border border-amber-400"></div>
                      <div className="w-3 h-3 rounded-full bg-emerald-400/30 border border-emerald-400"></div>
                  </div>
                </div>
    
                <div className="p-8 md:p-12 space-y-8">
                    {/* File Upload Zone */}
                    <div className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 group ${
                      file ? 'border-cyan-500 bg-cyan-50' : 'border-slate-300 hover:border-cyan-400 hover:bg-slate-50'
                    }`}>
                      <input
                        type="file"
                        accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                        className="hidden"
                        id="file-upload"
                      />
                      <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
                        {file ? (
                          <>
                            <div className="w-16 h-16 rounded-full bg-cyan-100 flex items-center justify-center text-cyan-600 border border-cyan-200">
                              <CheckCircle2 className="w-8 h-8" />
                            </div>
                            <div>
                              <p className="text-xl font-bold text-slate-900 break-all">{file.name}</p>
                              <p className="text-sm text-cyan-600 mt-1 font-semibold">File loaded successfully</p>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="w-16 h-16 rounded-full bg-slate-100 group-hover:bg-slate-200 flex items-center justify-center text-slate-400 transition-colors border border-slate-200">
                              <Upload className="w-8 h-8" />
                            </div>
                            <div>
                              <p className="text-xl font-bold text-slate-900">Drop NIfTI file or click to browse</p>
                              <p className="text-base text-slate-500 mt-2 font-medium">Supports .nii.gz 3D volumes</p>
                            </div>
                          </>
                        )}
                      </label>
                    </div>
    
                    {/* Analysis Button */}
                    <button 
                      onClick={handleUpload}
                      disabled={!file || loading}
                      className={`w-full py-5 rounded-xl font-bold text-xl transition-all ${
                        !file 
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' 
                          : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 hover:shadow-xl hover:shadow-cyan-500/30 text-white'
                      }`}
                    >
                      {loading ? (
                        <span className="flex items-center justify-center gap-2">
                          <Activity className="w-6 h-6 animate-spin" /> Analyzing Volume...
                        </span>
                      ) : "Run Analysis"}
                    </button>
    
                    {/* Results Display with Visualization */}
                    {result && result.predictions && (
                      <div className="space-y-6 pt-8 border-t border-slate-200 animate-in fade-in slide-in-from-bottom-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Classification Results</h3>
                            <span className="text-xs px-2 py-1 rounded bg-slate-100 text-slate-500 font-mono border border-slate-200">Model V5</span>
                        </div>

                        <div className="space-y-6">
                          {/* Negative (Healthy) Bar */}
                          <div className="space-y-2">
                            <div className="flex justify-between text-base">
                              <span className="text-slate-700 font-bold flex items-center gap-2"><CheckCircle2 className="w-5 h-5 text-emerald-500"/> Normal / Healthy</span>
                              <span className="font-mono text-emerald-600 font-black">{Math.round((result.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%</span>
                            </div>
                            <div className="h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                              <div className="h-full bg-emerald-500 transition-all duration-1000 shadow-sm" style={{width: `${(result.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}></div>
                            </div>
                          </div>

                          {/* Positive (Aneurysm) Bar */}
                          <div className="space-y-2">
                            <div className="flex justify-between text-base">
                              <span className="text-slate-700 font-bold flex items-center gap-2"><AlertCircle className="w-5 h-5 text-rose-500"/> Aneurysm Detected</span>
                              <span className="font-mono text-rose-600 font-black">{Math.round((result.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%</span>
                            </div>
                            <div className="h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                              <div className="h-full bg-rose-500 transition-all duration-1000 shadow-sm" style={{width: `${(result.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}></div>
                            </div>
                          </div>
                        </div>

                        {/* 3D Visualization Section */}
                        {result.scan_data && result.heatmap_data && (
                          <div className="pt-6 border-t border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-4">3D Visualization with Aneurysm Detection Heatmap</h3>
                            <NiftiViewer
                              scanData={result.scan_data}
                              heatmapData={result.heatmap_data}
                              predictedClass={result.predicted_class}
                              confidence={result.confidence}
                            />
                          </div>
                        )}

                        <div className="p-4 rounded-lg bg-blue-50 border border-blue-100 text-sm text-blue-800 leading-relaxed text-center font-medium">
                          Disclaimer: This tool is for research and experimental purposes only. Results must be clinically verified.
                        </div>
                      </div>
                    )}
                </div>
              </div>
            </div>
          </section>
    
          {/* Footer */}
          <footer className="py-12 text-center text-slate-500 text-sm border-t border-slate-200 bg-slate-50">
            <p>© 2025 NeuroScan Research. Powered by MedMNIST & PyTorch.</p>
          </footer>
        </main>
      );
    }