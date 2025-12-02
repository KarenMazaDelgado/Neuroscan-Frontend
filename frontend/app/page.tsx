'use client';
import { useState, useEffect } from 'react';
import { client } from "@gradio/client";
import { Upload, Activity, AlertCircle, CheckCircle2, ChevronRight, ShieldAlert, HeartPulse, Brain, Download, Database } from 'lucide-react';
import dynamic from 'next/dynamic';
import Image from 'next/image';

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

interface BatchResult extends AnalysisResult {
  filename: string;
  fileIndex: number;
}

export default function Home() {
    // Single upload state
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);

    // Batch upload state
    const [batchMode, setBatchMode] = useState(false);
    const [files, setFiles] = useState<File[]>([]);
    const [batchResults, setBatchResults] = useState<BatchResult[]>([]);
    const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
    const [selectedResult, setSelectedResult] = useState<BatchResult | null>(null);
    const [selectedForCompare, setSelectedForCompare] = useState<BatchResult[]>([]);

    // Compare mode state
    const [compareMode, setCompareMode] = useState(false);
    const [fileA, setFileA] = useState<File | null>(null);
    const [fileB, setFileB] = useState<File | null>(null);
    const [resultA, setResultA] = useState<AnalysisResult | null>(null);
    const [resultB, setResultB] = useState<AnalysisResult | null>(null);

    // Common state
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

    const handleBatchUpload = async () => {
        if (files.length === 0) return;
        setLoading(true);
        setBatchResults([]);
        setBatchProgress({ current: 0, total: files.length });

        const results: BatchResult[] = [];

        try {
            const app = await client(HF_SPACE_URL);
            console.log("Gradio client initialized for batch processing.");

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                console.log(`Processing file ${i + 1}/${files.length}: ${file.name}`);
                setBatchProgress({ current: i + 1, total: files.length });

                try {
                    // Add timeout wrapper (2 minutes per file)
                    const timeoutPromise = new Promise((_, reject) =>
                        setTimeout(() => reject(new Error('Timeout')), 120000)
                    );

                    const predictPromise = app.predict("/predict", [file]);

                    const response = await Promise.race([predictPromise, timeoutPromise]) as any;
                    console.log(`Response for ${file.name}:`, response);

                    if (response?.data?.[0]) {
                        const analysisResult: AnalysisResult = response.data[0];
                        results.push({
                            ...analysisResult,
                            filename: file.name,
                            fileIndex: i
                        });
                        // Update results incrementally
                        setBatchResults([...results]);
                        console.log(`✓ Successfully processed ${file.name}`);
                    } else {
                        console.error(`✗ Invalid response for ${file.name}`);
                    }
                } catch (error: any) {
                    console.error(`✗ Error processing ${file.name}:`, error.message || error);
                    // Continue with next file even if this one fails
                }

                // Small delay between requests to avoid overwhelming the backend
                if (i < files.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }

            console.log(`Batch complete: ${results.length}/${files.length} processed successfully`);

            if (results.length === 0) {
                alert('No files were processed successfully. Please check the console for errors.');
            } else if (results.length < files.length) {
                alert(`Processed ${results.length} out of ${files.length} files. Some files failed - check console for details.`);
            }

        } catch (error: any) {
            console.error("Batch processing error:", error);
            alert("Error initializing batch processing: " + (error.message || "Unknown error"));
        } finally {
            setLoading(false);
            setBatchProgress({ current: 0, total: 0 });
        }
    };

    const handleCompareUpload = async () => {
        if (!fileA || !fileB) return;
        setLoading(true);
        setResultA(null);
        setResultB(null);

        try {
            const app = await client(HF_SPACE_URL);
            console.log("Gradio client initialized for comparison.");

            // Process both files
            const [responseA, responseB] = await Promise.all([
                app.predict("/predict", [fileA]) as any,
                app.predict("/predict", [fileB]) as any
            ]);

            if (responseA?.data?.[0]) {
                setResultA(responseA.data[0]);
            }

            if (responseB?.data?.[0]) {
                setResultB(responseB.data[0]);
            }

            console.log("Comparison complete.");

        } catch (error: any) {
            console.error("Error during comparison:", error);
            alert("Comparison error: " + (error.message || "Unknown error"));
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
        <main className="min-h-screen font-sans selection:bg-blue-100">

          {/* ================= PROFESSIONAL NAVBAR ================= */}
          <div className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-slate-200 shadow-sm">
            <nav className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
              {/* Logo & Branding */}
              <div className="flex items-center gap-3 cursor-pointer" onClick={() => window.scrollTo(0,0)}>
                <div className="w-12 h-12 rounded bg-gradient-to-br from-blue-600 to-blue-700 flex items-center justify-center shadow-md">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {/* Brain outline */}
                    <path d="M12 3C8.5 3 6 5.5 6 8.5C6 9.5 6.3 10.4 6.8 11.2C6.3 11.8 6 12.6 6 13.5C6 15.4 7.3 17 9 17.5V19C9 20.1 9.9 21 11 21H13C14.1 21 15 20.1 15 19V17.5C16.7 17 18 15.4 18 13.5C18 12.6 17.7 11.8 17.2 11.2C17.7 10.4 18 9.5 18 8.5C18 5.5 15.5 3 12 3Z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    {/* Scan lines */}
                    <line x1="8" y1="10" x2="16" y2="10" stroke="white" strokeWidth="1.2" opacity="0.6"/>
                    <line x1="8" y1="13" x2="16" y2="13" stroke="white" strokeWidth="1.2" opacity="0.6"/>
                    <line x1="9" y1="16" x2="15" y2="16" stroke="white" strokeWidth="1.2" opacity="0.6"/>
                  </svg>
                </div>
                <div className="flex flex-col">
                  <span className="font-bold text-lg text-slate-900 leading-tight">
                    NeuroScan
                  </span>
                  <span className="text-xs text-slate-500 leading-tight">
                    Neuro Imaging AI
                  </span>
                </div>
              </div>

              {/* Links */}
              <div className="hidden md:flex items-center gap-1 text-sm font-semibold text-slate-600">
                <button onClick={() => scrollTo('impact')} className="hover:text-blue-700 hover:bg-blue-50 px-4 py-2 rounded transition-all">Impact</button>
                <button onClick={() => scrollTo('data-section')} className="hover:text-blue-700 hover:bg-blue-50 px-4 py-2 rounded transition-all">
                    Data & Validation
                </button>
              </div>

              {/* CTA Button */}
              <button
                onClick={() => scrollTo('diagnosis')}
                className="bg-blue-600 text-white hover:bg-blue-700 px-6 py-2.5 rounded text-sm font-bold transition-colors flex items-center gap-2 shadow-md"
              >
                Start Analysis <ChevronRight className="w-4 h-4" />
              </button>
            </nav>
          </div>
    
          {/* ================= HERO SECTION ================= */}
          <section className="relative pt-32 pb-24 px-6 overflow-hidden bg-gradient-to-br from-slate-50 via-white to-blue-50/30 border-b border-slate-200">
            {/* Background Medical Image with Overlay */}
            <div className="absolute inset-0 overflow-hidden">
              <img
                src="/images/brain-scan-hero.jpg"
                alt="Medical brain scan background"
                className="absolute inset-0 w-full h-full object-cover opacity-[0.08] z-0"
              />
              <div className="absolute inset-0 bg-gradient-to-br from-white via-white/95 to-blue-50/90 z-10"></div>
            </div>

            {/* Medical Grid Pattern Background */}
            <div
              className="absolute inset-0 opacity-[0.03] z-20"
              style={{
                backgroundImage: `linear-gradient(#1e40af 1px, transparent 1px), linear-gradient(90deg, #1e40af 1px, transparent 1px)`,
                backgroundSize: '50px 50px'
              }}
            ></div>

            {/* Large Decorative Medical Icons - Brain (Top Right) */}
            <div className="absolute top-20 right-10 opacity-[0.03] pointer-events-none z-20">
              <Brain className="w-64 h-64 text-blue-600" />
            </div>

            {/* Large Decorative Medical Icons - Activity (Bottom Left) */}
            <div className="absolute bottom-10 left-10 opacity-[0.03] pointer-events-none z-20">
              <Activity className="w-72 h-72 text-blue-600" />
            </div>

            {/* Subtle Background Gradients */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-blue-100/30 rounded-[100%] blur-[120px] -z-10 opacity-40"></div>

            <div className="max-w-5xl mx-auto text-center space-y-6 relative z-10">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded border border-emerald-200 bg-emerald-50 shadow-sm text-emerald-700 text-xs font-bold uppercase tracking-wider">
                <Activity className="w-3.5 h-3.5 animate-pulse" />
                V5 Model Online
              </div>

              <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-[1.15] text-slate-900">
                Brain Aneurysm Detection<br />
                <span className="text-blue-600">for Efficient Medical Triage</span>
              </h1>

              <p className="text-base md:text-lg text-slate-600 max-w-3xl mx-auto leading-relaxed">
                Advanced deep learning for efficient radiologist triage and accessible screening in under-resourced healthcare settings. AI-powered detection to flag potential aneurysms for clinical review and reduce diagnostic burden.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-6">
                <button onClick={() => scrollTo('diagnosis')} className="w-full sm:w-auto px-8 py-3.5 bg-blue-600 hover:bg-blue-700 text-white rounded font-bold transition-all shadow-lg">
                  Start Analysis
                </button>
                <button onClick={() => scrollTo('impact')} className="w-full sm:w-auto px-8 py-3.5 bg-white text-slate-700 border-2 border-slate-300 hover:border-blue-600 hover:text-blue-600 rounded font-bold transition-all">
                  View Impact
                </button>
              </div>
            </div>
          </section>
    
          {/* ================= IMPORTANCE / IMPACT SECTION ================= */}
          <section id="impact" className="py-24 px-6 bg-slate-50 relative overflow-hidden">
            {/* Background Medical Image with Strong Overlay */}
            <div className="absolute inset-0 overflow-hidden">
              <img
                src="/images/medical-team.jpg"
                alt="Medical professionals"
                className="absolute inset-0 w-full h-full object-cover opacity-[0.06] z-0"
              />
              <div className="absolute inset-0 bg-gradient-to-r from-slate-50 via-slate-50/98 to-slate-50/95 z-10"></div>
            </div>


            {/* Faded HeartPulse Icon Decoration */}
            <div className="absolute top-10 right-10 opacity-[0.03] pointer-events-none z-20">
              <HeartPulse className="w-56 h-56 text-blue-600" />
            </div>

            <div className="max-w-6xl mx-auto relative z-10">
              <div className="grid md:grid-cols-2 gap-20 items-center">

                <div className="space-y-10">
                  <h2 className="text-4xl font-bold text-slate-900">Impact</h2>
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
                        <h3 className="text-xl font-bold text-slate-900">The Cognitive Fatigue Problem</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          A landmark <a href="https://pubs.rsna.org/doi/abs/10.1148/radiol.2017170555" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline">2018 study analyzing 2.9 million radiology exams</a> found diagnostic errors peaked after <span className="font-black text-slate-900">10 hours</span> into shifts, with <span className="font-black text-slate-900">76% higher volume</span> on error-containing shifts.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-5">
                      <div className="mt-1 bg-blue-50 p-3 rounded-2xl h-fit border border-blue-100">
                        <Brain className="w-6 h-6 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">Efficient Triage for Radiologists</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          NeuroScan provides consistent automated pre-screening regardless of shift length or volume, flagging obvious cases so radiologists can focus their expertise on ambiguous findings and maintain diagnostic quality throughout their shifts.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-5">
                      <div className="mt-1 bg-emerald-50 p-3 rounded-2xl h-fit border border-emerald-100">
                        <HeartPulse className="w-6 h-6 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">Democratizing Healthcare Access</h3>
                        <p className="text-slate-600 mt-2 leading-relaxed text-lg">
                          NeuroScan is trained on lower-resolution data comparable to 1.5T MRI systems from the 2005-2012 era, allowing it to work with older equipment in rural hospitals and emerging <a href="https://www.nature.com/articles/s41467-021-25441-6" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline">portable low-field MRI systems</a>, making advanced screening accessible to underserved communities.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
    
                {/* Visual Stats (Gradient Background) */}
                <div className="relative p-4 pt-10">
                  {/* The subtle gradient effect (made larger and more central) */}
                  <div className="absolute inset-0 transform scale-150 bg-gradient-to-tr from-blue-200/25 via-rose-200/25 to-blue-300/25 rounded-full blur-[150px] -z-10 opacity-60"></div>

                  <div className="bg-white p-6 space-y-12 rounded-lg border border-slate-200 shadow-sm">
                    <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-slate-900 tracking-tight">6.5M</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">People in the US with unruptured aneurysms</div>
                    </div>

                      <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-rose-600 tracking-tight">30k</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">Ruptures occur each year</div>
                    </div>

                      <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-blue-700 tracking-tight">+226%</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">Error rate increase at high shift volumes (67-90 vs ≤19 studies)</div>
                    </div>

                      <div className="text-center space-y-2">
                      <div className="text-6xl md:text-7xl font-black text-emerald-600 tracking-tight">&lt;1:500k</div>
                      <div className="text-sm text-slate-500 font-bold uppercase tracking-widest">Typical radiologist ratio in sub-Saharan Africa</div>
                    </div>
                  </div>

                </div>
                
                {/* Sources Div, spanning both columns (or just placed after the grid) */}
              </div>
              <div className="pt-16 text-center">
                  <p className="text-sm text-slate-500 max-w-3xl mx-auto">
                    *Sources: 6.5 million prevalence and 30,000 annual ruptures based on common NIH/CDC estimates. Radiologist cognitive fatigue data from <a href="https://pubs.rsna.org/doi/abs/10.1148/radiol.2017170555" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline">Hanna et al., Radiology 2018</a>. Error rate volume correlation from <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11288559/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline">Ivanovic et al., AJNR 2024</a>. Sub-Saharan Africa radiologist shortage from <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3424787/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline">Journal of Clinical Imaging Science 2012</a>.
                  </p>
              </div>
            </div>
          </section>
    
          {/* ================= SCIENCE & DATA SECTION ================= */}
          <section id="data-section" className="py-24 px-6 bg-white relative overflow-hidden">
            {/* Faded Database Icon Decoration */}
            <div className="absolute top-20 left-10 opacity-[0.03] pointer-events-none">
              <Database className="w-64 h-64 text-blue-600" />
            </div>

            <div className="max-w-6xl mx-auto grid lg:grid-cols-12 gap-12 relative z-10">

              <div className="lg:col-span-7 space-y-12">
                <div>
                  <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-blue-100 rounded text-blue-700"><Database className="w-6 h-6" /></div>
                      <h2 className="text-3xl font-bold text-slate-900">Benchmarks & Data</h2>
                  </div>
                  <p className="text-slate-600 text-lg leading-relaxed">
                    Our V5 model achieved 83.72% sensitivity on low-resolution data (64×64×64 voxels), compared to ~95% sensitivity in clinical MRA studies using high-resolution scans (512×512×200+ voxels). While comparatively accurate given the significant resolution constraints, there remains an 11+ percentage point gap.
                  </p>
                </div>

                <div className="bg-slate-50 rounded-lg p-6 border border-slate-200 shadow-sm">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-700 font-semibold">NeuroScan V5 Sensitivity</span>
                      <span className="text-2xl font-black text-blue-600">83.72%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-700 font-semibold">Clinical MRA Studies</span>
                      <span className="text-2xl font-black text-slate-400">~95%</span>
                    </div>
                    <p className="text-sm text-slate-500 italic">Performance gap likely due to 64×64×64 vs. 512×512×200+ resolution difference</p>
                    <p className="text-sm text-slate-600 mt-3 pt-3 border-t border-slate-200">
                      <strong>Room for Improvement:</strong> Higher-resolution training data, architectural refinements, and expanded datasets could help close this gap. This tool should complement, not replace, clinical judgment.
                    </p>
                  </div>
                </div>
              </div>

              {/* Download Box */}
              <div className="lg:col-span-5 flex flex-col justify-center">
                <div className="bg-slate-50 p-8 rounded-lg border border-slate-200 shadow-md space-y-6">
                  <div className="text-center">
                    <div className="w-20 h-20 mx-auto bg-blue-50 rounded flex items-center justify-center text-blue-600 mb-4 border border-blue-100">
                      <Download className="w-10 h-10" />
                    </div>
                    <h3 className="text-2xl font-bold text-slate-900 mb-2">Validate the Findings</h3>
                    <p className="text-slate-600 text-sm">
                      We believe in transparency. Download anonymized test samples (NIfTI format) to validate our model using the diagnostic tool below.
                    </p>
                  </div>

                  {/* Batch Test Set Download */}
                  <div className="space-y-3">
                    <div className="text-left">
                      <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Complete Test Set</p>
                      <a
                        href="/test_samples_20.zip"
                        download
                        className="block w-full py-4 bg-blue-600 text-white hover:bg-blue-700 font-bold text-base rounded transition-colors shadow-md text-center"
                      >
                        Download 20 Test Scans (ZIP)
                      </a>
                      <p className="text-xs text-slate-500 mt-2 text-center">10 aneurysm + 10 normal scans</p>
                    </div>
                  </div>

                  {/* Individual Samples */}
                  <div className="space-y-3 pt-4 border-t border-slate-300">
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Individual Samples</p>
                    <div className="grid grid-cols-2 gap-2">
                      <a
                        href="/sample_aneurysm_001.nii"
                        download
                        className="px-3 py-2 bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 rounded text-xs font-semibold transition-colors text-center"
                      >
                        Aneurysm #1
                      </a>
                      <a
                        href="/sample_aneurysm_005.nii"
                        download
                        className="px-3 py-2 bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 rounded text-xs font-semibold transition-colors text-center"
                      >
                        Aneurysm #5
                      </a>
                      <a
                        href="/sample_normal_001.nii"
                        download
                        className="px-3 py-2 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 rounded text-xs font-semibold transition-colors text-center"
                      >
                        Normal #1
                      </a>
                      <a
                        href="/sample_normal_007.nii"
                        download
                        className="px-3 py-2 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 rounded text-xs font-semibold transition-colors text-center"
                      >
                        Normal #7
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
    
          {/* ================= APP INTERFACE ================= */}
          <section id="diagnosis" className="py-24 px-6 relative bg-slate-50 border-t border-slate-200 overflow-hidden">
            {/* Medical Grid Pattern Background */}
            <div
              className="absolute inset-0 opacity-[0.02]"
              style={{
                backgroundImage: `linear-gradient(#1e40af 1px, transparent 1px), linear-gradient(90deg, #1e40af 1px, transparent 1px)`,
                backgroundSize: '60px 60px'
              }}
            ></div>

            {/* Faded Brain Icon Decoration */}
            <div className="absolute bottom-10 right-10 opacity-[0.03] pointer-events-none">
              <Brain className="w-80 h-80 text-blue-600" />
            </div>

            <div className="max-w-4xl mx-auto relative z-10">

              <div className="bg-white rounded-lg overflow-hidden border-2 border-slate-200 shadow-lg">
                {/* Interface Header */}
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-6 border-b-2 border-blue-700 relative overflow-hidden">
                  {/* Animated Pulse Effect on Activity Icon */}
                  <div className="absolute top-1/2 right-6 -translate-y-1/2 opacity-10">
                    <Activity className="w-20 h-20 text-white animate-pulse" />
                  </div>

                  <div className="relative z-10">
                    <h2 className="text-2xl font-bold flex items-center gap-2 text-white">
                      <Activity className="w-6 h-6" />
                      Diagnostic Interface
                    </h2>
                    <p className="text-blue-100 text-sm mt-1">3D Volumetric Analysis System</p>
                  </div>
                </div>
    
                <div className="p-8 md:p-12 space-y-8">
                    {/* Single/Batch/Compare Mode Toggle */}
                    <div className="flex items-center justify-center gap-3 pb-4">
                      <button
                        onClick={() => {
                          setBatchMode(false);
                          setCompareMode(false);
                          setFiles([]);
                          setBatchResults([]);
                          setFileA(null);
                          setFileB(null);
                          setResultA(null);
                          setResultB(null);
                        }}
                        className={`px-6 py-3 rounded font-bold transition-all ${
                          !batchMode && !compareMode
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        Single
                      </button>
                      <button
                        onClick={() => {
                          setBatchMode(false);
                          setCompareMode(true);
                          setFile(null);
                          setResult(null);
                          setFiles([]);
                          setBatchResults([]);
                        }}
                        className={`px-6 py-3 rounded font-bold transition-all ${
                          compareMode
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        Compare
                      </button>
                      <button
                        onClick={() => {
                          setBatchMode(true);
                          setCompareMode(false);
                          setFile(null);
                          setResult(null);
                          setFileA(null);
                          setFileB(null);
                          setResultA(null);
                          setResultB(null);
                        }}
                        className={`px-6 py-3 rounded font-bold transition-all ${
                          batchMode
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        Batch
                      </button>
                    </div>

                    {/* File Upload Zone */}
                    {!batchMode && !compareMode ? (
                      // Single file upload
                      <div className={`border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 group ${
                        file ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
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
                              <div className="w-16 h-16 rounded bg-blue-100 flex items-center justify-center text-blue-600 border border-blue-200">
                                <CheckCircle2 className="w-8 h-8" />
                              </div>
                              <div>
                                <p className="text-xl font-bold text-slate-900 break-all">{file.name}</p>
                                <p className="text-sm text-blue-600 mt-1 font-semibold">File loaded successfully</p>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="w-16 h-16 rounded bg-slate-100 group-hover:bg-slate-200 flex items-center justify-center text-slate-400 transition-colors border border-slate-200">
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
                    ) : compareMode ? (
                      // Compare mode - two file uploads side by side
                      <div className="grid md:grid-cols-2 gap-4">
                        {/* File A Upload */}
                        <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 group ${
                          fileA ? 'border-blue-600 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
                        }`}>
                          <input
                            type="file"
                            accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream"
                            onChange={(e) => setFileA(e.target.files?.[0] || null)}
                            className="hidden"
                            id="file-upload-a"
                          />
                          <label htmlFor="file-upload-a" className="cursor-pointer flex flex-col items-center gap-3">
                            {fileA ? (
                              <>
                                <div className="w-12 h-12 rounded bg-blue-100 flex items-center justify-center text-blue-600 border border-blue-200">
                                  <CheckCircle2 className="w-6 h-6" />
                                </div>
                                <div>
                                  <p className="text-sm font-bold text-blue-600 mb-1">Scan A</p>
                                  <p className="text-base font-bold text-slate-900 break-all">{fileA.name}</p>
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="w-12 h-12 rounded bg-slate-100 group-hover:bg-slate-200 flex items-center justify-center text-slate-400 transition-colors border border-slate-200">
                                  <Upload className="w-6 h-6" />
                                </div>
                                <div>
                                  <p className="text-sm font-bold text-slate-600 mb-1">Scan A</p>
                                  <p className="text-base font-bold text-slate-900">Upload first scan</p>
                                </div>
                              </>
                            )}
                          </label>
                        </div>

                        {/* File B Upload */}
                        <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 group ${
                          fileB ? 'border-purple-500 bg-purple-50' : 'border-slate-300 hover:border-purple-400 hover:bg-slate-50'
                        }`}>
                          <input
                            type="file"
                            accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream"
                            onChange={(e) => setFileB(e.target.files?.[0] || null)}
                            className="hidden"
                            id="file-upload-b"
                          />
                          <label htmlFor="file-upload-b" className="cursor-pointer flex flex-col items-center gap-3">
                            {fileB ? (
                              <>
                                <div className="w-12 h-12 rounded bg-purple-100 flex items-center justify-center text-purple-600 border border-purple-200">
                                  <CheckCircle2 className="w-6 h-6" />
                                </div>
                                <div>
                                  <p className="text-sm font-bold text-purple-600 mb-1">Scan B</p>
                                  <p className="text-base font-bold text-slate-900 break-all">{fileB.name}</p>
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="w-12 h-12 rounded bg-slate-100 group-hover:bg-slate-200 flex items-center justify-center text-slate-400 transition-colors border border-slate-200">
                                  <Upload className="w-6 h-6" />
                                </div>
                                <div>
                                  <p className="text-sm font-bold text-slate-600 mb-1">Scan B</p>
                                  <p className="text-base font-bold text-slate-900">Upload second scan</p>
                                </div>
                              </>
                            )}
                          </label>
                        </div>
                      </div>
                    ) : batchMode ? (
                      // Batch file upload
                      <div className={`border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 group ${
                        files.length > 0 ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
                      }`}>
                        <input
                          type="file"
                          accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream"
                          onChange={(e) => setFiles(Array.from(e.target.files || []))}
                          className="hidden"
                          id="batch-file-upload"
                          multiple
                        />
                        <label htmlFor="batch-file-upload" className="cursor-pointer flex flex-col items-center gap-4">
                          {files.length > 0 ? (
                            <>
                              <div className="w-16 h-16 rounded bg-blue-100 flex items-center justify-center text-blue-600 border border-blue-200">
                                <CheckCircle2 className="w-8 h-8" />
                              </div>
                              <div>
                                <p className="text-xl font-bold text-slate-900">{files.length} files selected</p>
                                <p className="text-sm text-blue-600 mt-1 font-semibold">Ready for batch processing</p>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="w-16 h-16 rounded bg-slate-100 group-hover:bg-slate-200 flex items-center justify-center text-slate-400 transition-colors border border-slate-200">
                                <Upload className="w-8 h-8" />
                              </div>
                              <div>
                                <p className="text-xl font-bold text-slate-900">Drop multiple NIfTI files or click to browse</p>
                                <p className="text-base text-slate-500 mt-2 font-medium">Supports .nii.gz 3D volumes</p>
                              </div>
                            </>
                          )}
                        </label>
                      </div>
                    ) : null}

                    {/* Analysis Button */}
                    <button
                      onClick={compareMode ? handleCompareUpload : (batchMode ? handleBatchUpload : handleUpload)}
                      disabled={
                        compareMode
                          ? (!fileA || !fileB || loading)
                          : (batchMode ? (files.length === 0 || loading) : (!file || loading))
                      }
                      className={`w-full py-5 rounded font-bold text-xl transition-all ${
                        (compareMode ? (!fileA || !fileB) : (batchMode ? files.length === 0 : !file))
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                          : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg text-white'
                      }`}
                    >
                      {loading ? (
                        <span className="flex items-center justify-center gap-2">
                          <Activity className="w-6 h-6 animate-spin" />
                          {compareMode
                            ? 'Comparing Scans...'
                            : (batchMode
                              ? `Processing ${batchProgress.current} of ${batchProgress.total}...`
                              : 'Analyzing Volume...'
                            )
                          }
                        </span>
                      ) : compareMode
                        ? 'Compare Scans'
                        : (batchMode ? `Run Batch Analysis (${files.length} files)` : "Run Analysis")
                      }
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

                    {/* Compare Mode Side-by-Side Results */}
                    {compareMode && resultA && resultB && (
                      <div className="space-y-6 pt-8 border-t border-slate-200 animate-in fade-in slide-in-from-bottom-4">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider text-center">Side-by-Side Comparison</h3>

                        <div className="grid md:grid-cols-2 gap-6">
                          {/* Scan A Results */}
                          <div className="space-y-4 p-6 bg-blue-50 rounded-lg border-2 border-blue-200">
                            <div className="text-center mb-4">
                              <p className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-1">Scan A</p>
                              <p className="text-sm text-slate-600 font-mono truncate">{fileA?.name}</p>
                            </div>

                            {/* Classification Bars */}
                            <div className="space-y-3">
                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-slate-700 font-bold flex items-center gap-1">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500"/> Normal
                                  </span>
                                  <span className="font-mono text-emerald-600 font-black text-sm">
                                    {Math.round((resultA.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%
                                  </span>
                                </div>
                                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-emerald-500 transition-all duration-1000"
                                    style={{width: `${(resultA.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}
                                  ></div>
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-slate-700 font-bold flex items-center gap-1">
                                    <AlertCircle className="w-4 h-4 text-rose-500"/> Aneurysm
                                  </span>
                                  <span className="font-mono text-rose-600 font-black text-sm">
                                    {Math.round((resultA.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%
                                  </span>
                                </div>
                                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-rose-500 transition-all duration-1000"
                                    style={{width: `${(resultA.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}
                                  ></div>
                                </div>
                              </div>
                            </div>

                            {/* 3D Visualization A */}
                            {resultA.scan_data && resultA.heatmap_data && (
                              <div className="mt-4">
                                <NiftiViewer
                                  scanData={resultA.scan_data}
                                  heatmapData={resultA.heatmap_data}
                                  predictedClass={resultA.predicted_class}
                                  confidence={resultA.confidence}
                                  showExplanations={false}
                                />
                              </div>
                            )}
                          </div>

                          {/* Scan B Results */}
                          <div className="space-y-4 p-6 bg-purple-50 rounded-lg border-2 border-purple-200">
                            <div className="text-center mb-4">
                              <p className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-1">Scan B</p>
                              <p className="text-sm text-slate-600 font-mono truncate">{fileB?.name}</p>
                            </div>

                            {/* Classification Bars */}
                            <div className="space-y-3">
                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-slate-700 font-bold flex items-center gap-1">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500"/> Normal
                                  </span>
                                  <span className="font-mono text-emerald-600 font-black text-sm">
                                    {Math.round((resultB.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%
                                  </span>
                                </div>
                                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-emerald-500 transition-all duration-1000"
                                    style={{width: `${(resultB.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}
                                  ></div>
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className="text-slate-700 font-bold flex items-center gap-1">
                                    <AlertCircle className="w-4 h-4 text-rose-500"/> Aneurysm
                                  </span>
                                  <span className="font-mono text-rose-600 font-black text-sm">
                                    {Math.round((resultB.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%
                                  </span>
                                </div>
                                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-rose-500 transition-all duration-1000"
                                    style={{width: `${(resultB.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}
                                  ></div>
                                </div>
                              </div>
                            </div>

                            {/* 3D Visualization B */}
                            {resultB.scan_data && resultB.heatmap_data && (
                              <div className="mt-4">
                                <NiftiViewer
                                  scanData={resultB.scan_data}
                                  heatmapData={resultB.heatmap_data}
                                  predictedClass={resultB.predicted_class}
                                  confidence={resultB.confidence}
                                  showExplanations={false}
                                />
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Single explanation section at bottom */}
                        <div className="space-y-4">
                          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                            <p className="text-sm font-bold text-slate-700 mb-2">Understanding the Views</p>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-slate-600">
                              <div>
                                <p className="font-semibold text-slate-800">Top Left: Side View</p>
                                <p className="text-slate-500">Sagittal (L↔R)</p>
                              </div>
                              <div>
                                <p className="font-semibold text-slate-800">Top Right: Front View</p>
                                <p className="text-slate-500">Coronal (F↔B)</p>
                              </div>
                              <div>
                                <p className="font-semibold text-slate-800">Bottom Left: Top View</p>
                                <p className="text-slate-500">Axial (T↔B)</p>
                              </div>
                              <div>
                                <p className="font-semibold text-slate-800">Bottom Right: 3D View</p>
                                <p className="text-slate-500">Volume Rendering</p>
                              </div>
                            </div>
                          </div>

                          <div className="bg-white p-4 rounded-lg border border-slate-200">
                            <p className="text-sm font-bold text-slate-700 mb-3">Aneurysm Detection Heatmap</p>
                            <div className="flex items-center gap-4 mb-3">
                              <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-gray-400 rounded"></div>
                                <span className="text-sm text-slate-600">Vessel Tissue</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-4 rounded" style={{background: 'linear-gradient(to right, #FFFF00, #FF8C00, #FF0000)'}}></div>
                                <span className="text-sm text-slate-600">Suspicion Level (Low → High)</span>
                              </div>
                            </div>
                            <div className="text-xs text-slate-600 space-y-1 bg-slate-50 p-3 rounded">
                              <p>• <strong>Yellow:</strong> Low suspicion</p>
                              <p>• <strong>Orange:</strong> Moderate suspicion</p>
                              <p>• <strong>Red:</strong> High suspicion - likely aneurysm</p>
                            </div>
                          </div>

                          <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                            <p className="text-sm font-bold text-blue-900 mb-2">Viewer Controls</p>
                            <ul className="text-xs text-blue-800 space-y-1">
                              <li>• <strong>Click:</strong> Move crosshair</li>
                              <li>• <strong>Right Click + Drag:</strong> Adjust brightness/contrast</li>
                              <li>• <strong>Scroll:</strong> Zoom</li>
                            </ul>
                          </div>
                        </div>

                        <div className="p-4 rounded-lg bg-blue-50 border border-blue-100 text-sm text-blue-800 leading-relaxed text-center font-medium">
                          Disclaimer: This tool is for research and experimental purposes only. Results must be clinically verified.
                        </div>
                      </div>
                    )}

                    {/* Batch Results Grid */}
                    {batchMode && batchResults.length > 0 && (
                      <div className="space-y-6 pt-8 border-t border-slate-200 animate-in fade-in slide-in-from-bottom-4">
                        <div className="flex items-center justify-between flex-wrap gap-3">
                          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">
                            Batch Results ({batchResults.length} files)
                          </h3>
                          <div className="flex items-center gap-3">
                            {selectedForCompare.length === 2 && (
                              <button
                                onClick={() => {
                                  // Open comparison modal with the two selected scans
                                  setSelectedResult(null); // Clear single view
                                }}
                                className="px-4 py-2 bg-blue-600 text-white rounded text-sm font-bold hover:bg-blue-700 transition-all flex items-center gap-2"
                              >
                                Compare Selected ({selectedForCompare.length})
                              </button>
                            )}
                            <button
                              onClick={() => {
                                const aneurysmFiles = batchResults.filter(r => r.predicted_class === 'Aneurysm');
                                const normalFiles = batchResults.filter(r => r.predicted_class === 'Normal');
                                alert(`Aneurysm: ${aneurysmFiles.length} files\nNormal: ${normalFiles.length} files\n\nDownload feature coming soon!`);
                              }}
                              className="px-4 py-2 bg-slate-700 text-white rounded text-sm font-bold hover:bg-slate-800 transition-all flex items-center gap-2"
                            >
                              <Download className="w-4 h-4" />
                              Export Results
                            </button>
                          </div>
                        </div>

                        {/* Selection Instructions */}
                        {selectedForCompare.length > 0 && (
                          <div className="flex items-center justify-center gap-2 text-sm bg-blue-50 p-3 rounded border border-blue-200">
                            <span className="text-blue-700 font-semibold">
                              {selectedForCompare.length === 1
                                ? 'Select one more scan to compare'
                                : 'Click "Compare Selected" to view side-by-side'}
                            </span>
                            <button
                              onClick={() => setSelectedForCompare([])}
                              className="text-blue-600 hover:text-blue-800 underline font-bold"
                            >
                              Clear Selection
                            </button>
                          </div>
                        )}

                        {/* Color Legend */}
                        <div className="flex items-center justify-center gap-6 text-sm bg-slate-50 p-4 rounded-lg border border-slate-200">
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-emerald-500"></div>
                            <span className="text-slate-600 font-semibold">Normal (Confidence &gt; 70%)</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-amber-500"></div>
                            <span className="text-slate-600 font-semibold">Uncertain (50-70%)</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-rose-500"></div>
                            <span className="text-slate-600 font-semibold">Aneurysm (Confidence &gt; 70%)</span>
                          </div>
                        </div>

                        {/* Grid View - Compact */}
                        <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                          {batchResults.map((result, idx) => {
                            const isAneurysm = result.predicted_class === 'Aneurysm';
                            const confidence = result.confidence * 100;
                            const isSelected = selectedForCompare.some(s => s.fileIndex === result.fileIndex);

                            // Determine color based on prediction and confidence
                            let colorClass = 'border-amber-400 bg-amber-50';
                            if (confidence > 70) {
                              colorClass = isAneurysm
                                ? 'border-rose-400 bg-rose-50'
                                : 'border-emerald-400 bg-emerald-50';
                            }

                            // Add selection styling
                            if (isSelected) {
                              colorClass = 'border-blue-600 bg-blue-100 ring-2 ring-blue-200';
                            }

                            return (
                              <div key={idx} className="relative">
                                <button
                                  onClick={() => setSelectedResult(result)}
                                  onContextMenu={(e) => {
                                    e.preventDefault();
                                    if (isSelected) {
                                      setSelectedForCompare(selectedForCompare.filter(s => s.fileIndex !== result.fileIndex));
                                    } else if (selectedForCompare.length < 2) {
                                      setSelectedForCompare([...selectedForCompare, result]);
                                    }
                                  }}
                                  className={`w-full p-2 rounded-lg border-2 ${colorClass} hover:shadow-md transition-all text-left relative`}
                                  title={result.filename}
                                >
                                  {/* Selection Checkbox - Smaller */}
                                  <div
                                    className="absolute top-1 right-1 z-10"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (isSelected) {
                                        setSelectedForCompare(selectedForCompare.filter(s => s.fileIndex !== result.fileIndex));
                                      } else if (selectedForCompare.length < 2) {
                                        setSelectedForCompare([...selectedForCompare, result]);
                                      }
                                    }}
                                  >
                                    <div className={`w-4 h-4 rounded border flex items-center justify-center transition-all ${
                                      isSelected
                                        ? 'bg-blue-600 border-blue-600'
                                        : 'bg-white border-slate-300 hover:border-blue-400'
                                    }`}>
                                      {isSelected && <CheckCircle2 className="w-3 h-3 text-white" />}
                                    </div>
                                  </div>

                                  <div className="pr-5">
                                    <p className="text-xs text-slate-500 font-mono truncate mb-1">{result.filename}</p>
                                    <p className={`text-lg font-black ${
                                      isAneurysm ? 'text-rose-700' : 'text-emerald-700'
                                    }`}>
                                      {Math.round(confidence)}%
                                    </p>
                                  </div>
                                </button>
                              </div>
                            );
                          })}
                        </div>

                        <div className="p-4 rounded-lg bg-blue-50 border border-blue-100 text-sm text-blue-800 leading-relaxed text-center font-medium">
                          Disclaimer: This tool is for research and experimental purposes only. Results must be clinically verified.
                        </div>
                      </div>
                    )}

                    {/* Comparison Modal for Selected Batch Scans */}
                    {selectedForCompare.length === 2 && !selectedResult && (
                      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedForCompare([])}>
                        <div className="bg-white rounded-lg max-w-7xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
                          {/* Modal Header */}
                          <div className="sticky top-0 bg-white border-b border-slate-200 p-6 rounded-t-lg flex items-center justify-between z-10">
                            <div>
                              <h3 className="text-2xl font-bold text-slate-900">Scan Comparison</h3>
                              <p className="text-sm text-slate-500 mt-1">Side-by-Side Analysis</p>
                            </div>
                            <button
                              onClick={() => setSelectedForCompare([])}
                              className="w-10 h-10 rounded bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors"
                            >
                              <span className="text-2xl text-slate-600">×</span>
                            </button>
                          </div>

                          {/* Modal Content - Side by Side */}
                          <div className="p-6">
                            <div className="grid md:grid-cols-2 gap-6">
                              {/* First Selected Scan */}
                              <div className="space-y-4 p-6 bg-blue-50 rounded-lg border-2 border-blue-200">
                                <div className="text-center mb-4">
                                  <p className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-1">Scan 1</p>
                                  <p className="text-sm text-slate-600 font-mono truncate">{selectedForCompare[0].filename}</p>
                                </div>

                                <div className="space-y-3">
                                  <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-slate-700 font-bold flex items-center gap-1">
                                        <CheckCircle2 className="w-4 h-4 text-emerald-500"/> Normal
                                      </span>
                                      <span className="font-mono text-emerald-600 font-black text-sm">
                                        {Math.round((selectedForCompare[0].predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%
                                      </span>
                                    </div>
                                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-emerald-500 transition-all duration-1000"
                                        style={{width: `${(selectedForCompare[0].predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}
                                      ></div>
                                    </div>
                                  </div>

                                  <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-slate-700 font-bold flex items-center gap-1">
                                        <AlertCircle className="w-4 h-4 text-rose-500"/> Aneurysm
                                      </span>
                                      <span className="font-mono text-rose-600 font-black text-sm">
                                        {Math.round((selectedForCompare[0].predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%
                                      </span>
                                    </div>
                                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-rose-500 transition-all duration-1000"
                                        style={{width: `${(selectedForCompare[0].predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}
                                      ></div>
                                    </div>
                                  </div>
                                </div>

                                {selectedForCompare[0].scan_data && selectedForCompare[0].heatmap_data && (
                                  <div className="mt-4">
                                    <NiftiViewer
                                      scanData={selectedForCompare[0].scan_data}
                                      heatmapData={selectedForCompare[0].heatmap_data}
                                      predictedClass={selectedForCompare[0].predicted_class}
                                      confidence={selectedForCompare[0].confidence}
                                      showExplanations={false}
                                    />
                                  </div>
                                )}
                              </div>

                              {/* Second Selected Scan */}
                              <div className="space-y-4 p-6 bg-purple-50 rounded-lg border-2 border-purple-200">
                                <div className="text-center mb-4">
                                  <p className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-1">Scan 2</p>
                                  <p className="text-sm text-slate-600 font-mono truncate">{selectedForCompare[1].filename}</p>
                                </div>

                                <div className="space-y-3">
                                  <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-slate-700 font-bold flex items-center gap-1">
                                        <CheckCircle2 className="w-4 h-4 text-emerald-500"/> Normal
                                      </span>
                                      <span className="font-mono text-emerald-600 font-black text-sm">
                                        {Math.round((selectedForCompare[1].predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%
                                      </span>
                                    </div>
                                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-emerald-500 transition-all duration-1000"
                                        style={{width: `${(selectedForCompare[1].predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}
                                      ></div>
                                    </div>
                                  </div>

                                  <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-slate-700 font-bold flex items-center gap-1">
                                        <AlertCircle className="w-4 h-4 text-rose-500"/> Aneurysm
                                      </span>
                                      <span className="font-mono text-rose-600 font-black text-sm">
                                        {Math.round((selectedForCompare[1].predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%
                                      </span>
                                    </div>
                                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-rose-500 transition-all duration-1000"
                                        style={{width: `${(selectedForCompare[1].predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}
                                      ></div>
                                    </div>
                                  </div>
                                </div>

                                {selectedForCompare[1].scan_data && selectedForCompare[1].heatmap_data && (
                                  <div className="mt-4">
                                    <NiftiViewer
                                      scanData={selectedForCompare[1].scan_data}
                                      heatmapData={selectedForCompare[1].heatmap_data}
                                      predictedClass={selectedForCompare[1].predicted_class}
                                      confidence={selectedForCompare[1].confidence}
                                      showExplanations={false}
                                    />
                                  </div>
                                )}
                              </div>
                            </div>

                            {/* Single explanation section at bottom */}
                            <div className="space-y-4 mt-6">
                              <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                                <p className="text-sm font-bold text-slate-700 mb-2">Understanding the Views</p>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-slate-600">
                                  <div>
                                    <p className="font-semibold text-slate-800">Top Left: Side View</p>
                                    <p className="text-slate-500">Sagittal (L↔R)</p>
                                  </div>
                                  <div>
                                    <p className="font-semibold text-slate-800">Top Right: Front View</p>
                                    <p className="text-slate-500">Coronal (F↔B)</p>
                                  </div>
                                  <div>
                                    <p className="font-semibold text-slate-800">Bottom Left: Top View</p>
                                    <p className="text-slate-500">Axial (T↔B)</p>
                                  </div>
                                  <div>
                                    <p className="font-semibold text-slate-800">Bottom Right: 3D View</p>
                                    <p className="text-slate-500">Volume Rendering</p>
                                  </div>
                                </div>
                              </div>

                              <div className="bg-white p-4 rounded-lg border border-slate-200">
                                <p className="text-sm font-bold text-slate-700 mb-3">Aneurysm Detection Heatmap</p>
                                <div className="flex items-center gap-4 mb-3">
                                  <div className="flex items-center gap-2">
                                    <div className="w-4 h-4 bg-gray-400 rounded"></div>
                                    <span className="text-sm text-slate-600">Vessel Tissue</span>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <div className="w-16 h-4 rounded" style={{background: 'linear-gradient(to right, #FFFF00, #FF8C00, #FF0000)'}}></div>
                                    <span className="text-sm text-slate-600">Suspicion Level</span>
                                  </div>
                                </div>
                                <div className="text-xs text-slate-600 space-y-1 bg-slate-50 p-3 rounded">
                                  <p>• <strong>Yellow:</strong> Low suspicion</p>
                                  <p>• <strong>Orange:</strong> Moderate suspicion</p>
                                  <p>• <strong>Red:</strong> High suspicion</p>
                                </div>
                              </div>

                              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                                <p className="text-sm font-bold text-blue-900 mb-2">Viewer Controls</p>
                                <ul className="text-xs text-blue-800 space-y-1">
                                  <li>• <strong>Click:</strong> Move crosshair</li>
                                  <li>• <strong>Right Click + Drag:</strong> Adjust brightness/contrast</li>
                                  <li>• <strong>Scroll:</strong> Zoom</li>
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Modal for Batch Result Detail View */}
                    {selectedResult && (
                      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedResult(null)}>
                        <div className="bg-white rounded-lg max-w-5xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
                          {/* Modal Header */}
                          <div className="sticky top-0 bg-white border-b border-slate-200 p-6 rounded-t-lg flex items-center justify-between z-10">
                            <div>
                              <h3 className="text-2xl font-bold text-slate-900">{selectedResult.filename}</h3>
                              <p className="text-sm text-slate-500 mt-1">Detailed Analysis View</p>
                            </div>
                            <button
                              onClick={() => setSelectedResult(null)}
                              className="w-10 h-10 rounded bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors"
                            >
                              <span className="text-2xl text-slate-600">×</span>
                            </button>
                          </div>

                          {/* Modal Content */}
                          <div className="p-6 space-y-6">
                            {/* Classification Results */}
                            <div className="space-y-4">
                              <h4 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Classification Results</h4>
                              <div className="space-y-4">
                                {/* Normal Bar */}
                                <div className="space-y-2">
                                  <div className="flex justify-between text-base">
                                    <span className="text-slate-700 font-bold flex items-center gap-2">
                                      <CheckCircle2 className="w-5 h-5 text-emerald-500"/> Normal / Healthy
                                    </span>
                                    <span className="font-mono text-emerald-600 font-black">
                                      {Math.round((selectedResult.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100)}%
                                    </span>
                                  </div>
                                  <div className="h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                                    <div
                                      className="h-full bg-emerald-500 transition-all duration-1000"
                                      style={{width: `${(selectedResult.predictions.find(p => p[0] === "Normal")?.[1] || 0) * 100}%`}}
                                    ></div>
                                  </div>
                                </div>

                                {/* Aneurysm Bar */}
                                <div className="space-y-2">
                                  <div className="flex justify-between text-base">
                                    <span className="text-slate-700 font-bold flex items-center gap-2">
                                      <AlertCircle className="w-5 h-5 text-rose-500"/> Aneurysm Detected
                                    </span>
                                    <span className="font-mono text-rose-600 font-black">
                                      {Math.round((selectedResult.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100)}%
                                    </span>
                                  </div>
                                  <div className="h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                                    <div
                                      className="h-full bg-rose-500 transition-all duration-1000"
                                      style={{width: `${(selectedResult.predictions.find(p => p[0] === "Aneurysm")?.[1] || 0) * 100}%`}}
                                    ></div>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* 3D Visualization */}
                            {selectedResult.scan_data && selectedResult.heatmap_data && (
                              <div className="pt-6 border-t border-slate-200">
                                <h4 className="text-lg font-bold text-slate-900 mb-4">3D Visualization with Aneurysm Detection Heatmap</h4>
                                <NiftiViewer
                                  scanData={selectedResult.scan_data}
                                  heatmapData={selectedResult.heatmap_data}
                                  predictedClass={selectedResult.predicted_class}
                                  confidence={selectedResult.confidence}
                                />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                </div>
              </div>
            </div>
          </section>
    
          {/* Footer */}
          <footer className="py-12 text-center text-slate-500 text-sm border-t border-slate-200 bg-slate-50 relative overflow-hidden">
            {/* Subtle Medical Grid */}
            <div
              className="absolute inset-0 opacity-[0.015]"
              style={{
                backgroundImage: `linear-gradient(#1e40af 1px, transparent 1px), linear-gradient(90deg, #1e40af 1px, transparent 1px)`,
                backgroundSize: '40px 40px'
              }}
            ></div>

            <p className="relative z-10">© 2025 NeuroScan Research. Powered by MedMNIST & PyTorch.</p>
          </footer>
        </main>
      );
    }