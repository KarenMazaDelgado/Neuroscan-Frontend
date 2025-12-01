'use client';

import { useEffect, useRef } from 'react';
import { Niivue, NVImage } from '@niivue/niivue';

interface NiftiViewerProps {
  scanData: string;  // base64 encoded NIfTI
  heatmapData: string;  // base64 encoded heatmap NIfTI
  predictedClass: string;
  confidence: number;
}

export default function NiftiViewer({ scanData, heatmapData, predictedClass, confidence }: NiftiViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize Niivue
    const nv = new Niivue({
      dragAndDropEnabled: false,
      backColor: [0, 0, 0, 1],
      show3Dcrosshair: true,
    });

    nv.attachToCanvas(canvasRef.current);
    nvRef.current = nv;

    // Load images
    loadImages();

    return () => {
      // Cleanup
      if (nvRef.current) {
        nvRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (nvRef.current && scanData && heatmapData) {
      loadImages();
    }
  }, [scanData, heatmapData]);

  const loadImages = async () => {
    if (!nvRef.current || !scanData || !heatmapData) return;

    try {
      // Convert base64 to blob
      const scanBlob = base64ToBlob(scanData);
      const heatmapBlob = base64ToBlob(heatmapData);

      // Create object URLs from blobs
      const scanUrl = URL.createObjectURL(scanBlob);
      const heatmapUrl = URL.createObjectURL(heatmapBlob);

      // Load volumes using URLs with filenames
      await nvRef.current.loadVolumes([
        { url: scanUrl, name: 'scan.nii' },
        { url: heatmapUrl, name: 'heatmap.nii' }
      ]);

      console.log('Loaded volumes:', nvRef.current.volumes.length);
      console.log('Volume 0 (scan):', nvRef.current.volumes[0]);
      console.log('Volume 1 (heatmap):', nvRef.current.volumes[1]);

      // Set heatmap overlay properties after loading
      if (nvRef.current.volumes.length > 1) {
        const heatmap = nvRef.current.volumes[1];

        // Use 'warm' colormap (yellow-orange-red) which is better for heatmaps
        heatmap.colormap = 'warm';
        heatmap.opacity = 1.0;  // Full opacity for visibility

        // Threshold the heatmap to only show high-attention regions (top 15%)
        if (heatmap.global_max !== undefined) {
          const threshold = heatmap.global_max * 0.15;
          heatmap.cal_min = threshold;
          heatmap.cal_max = heatmap.global_max;

          console.log('Heatmap properties:', {
            opacity: heatmap.opacity,
            colormap: heatmap.colormap,
            cal_min: heatmap.cal_min,
            cal_max: heatmap.cal_max,
            global_min: heatmap.global_min,
            global_max: heatmap.global_max,
            threshold: threshold
          });
        }

        nvRef.current.updateGLVolume();
      }

      // Set view to show all 3 planes and change crosshair color
      nvRef.current.setSliceType(nvRef.current.sliceTypeMultiplanar);
      nvRef.current.opts.crosshairColor = [0, 1, 0, 1];  // Green crosshairs instead of red

      // Clean up object URLs to free memory
      URL.revokeObjectURL(scanUrl);
      URL.revokeObjectURL(heatmapUrl);

    } catch (error) {
      console.error('Error loading NIfTI images:', error);
    }
  };

  const base64ToBlob = (base64: string): Blob => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: 'application/octet-stream' });
  };

  return (
    <div className="space-y-4">
      {/* Info Card */}
      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-500 font-semibold">Predicted Classification</p>
            <p className={`text-2xl font-bold ${predictedClass === 'Aneurysm' ? 'text-rose-600' : 'text-emerald-600'}`}>
              {predictedClass}
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-slate-500 font-semibold">Confidence</p>
            <p className="text-2xl font-bold text-slate-900">{Math.round(confidence * 100)}%</p>
          </div>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="bg-black rounded-xl overflow-hidden border-2 border-slate-200 shadow-xl">
        <canvas
          ref={canvasRef}
          className="w-full"
          style={{ minHeight: '500px' }}
        />
      </div>

      {/* View Explanation */}
      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
        <p className="text-sm font-bold text-slate-700 mb-2">Understanding the Views</p>
        <div className="grid grid-cols-3 gap-3 text-xs text-slate-600">
          <div>
            <p className="font-semibold text-slate-800">Left Panel: Side View</p>
            <p className="text-slate-500">Sagittal slice (L↔R)</p>
          </div>
          <div>
            <p className="font-semibold text-slate-800">Middle Panel: Front View</p>
            <p className="text-slate-500">Coronal slice (F↔B)</p>
          </div>
          <div>
            <p className="font-semibold text-slate-800">Right Panel: Top View</p>
            <p className="text-slate-500">Axial slice (T↔B)</p>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          <strong>Note:</strong> Each sample shows a 64×64×64 voxel segment containing a blood vessel.
          Some samples may appear faint or empty if the vessel segment is small or has low contrast.
        </p>
      </div>

      {/* Legend */}
      <div className="bg-white p-4 rounded-xl border border-slate-200">
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
          <p>• <strong>Yellow overlay:</strong> Low suspicion - minor anomalies detected</p>
          <p>• <strong>Orange overlay:</strong> Moderate suspicion - potential abnormalities</p>
          <p>• <strong>Red overlay:</strong> High suspicion - likely aneurysm location</p>
          <p>• <strong>No overlay:</strong> No significant abnormalities detected in the vessel</p>
        </div>
      </div>

      {/* Controls Help */}
      <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
        <p className="text-sm font-bold text-blue-900 mb-2">Viewer Controls</p>
        <ul className="text-xs text-blue-800 space-y-1">
          <li>• <strong>Click:</strong> Move crosshair and view different slice positions</li>
          <li>• <strong>Right Click + Drag:</strong> Adjust brightness and contrast</li>
          <li>• <strong>Scroll / Two-finger swipe:</strong> Zoom in and out</li>
          <li>• <strong>Double Click:</strong> Reset view</li>
        </ul>
      </div>
    </div>
  );
}
