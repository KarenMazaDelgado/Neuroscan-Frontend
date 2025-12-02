import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Ensure the rewrites function is defined correctly within the configuration object
  async rewrites() {
    return [
      {
        // This is the local path used in your page.tsx: const HF_SPACE_URL = "/api/gradio-proxy/"; 
        source: '/api/gradio-proxy/:path*',
        
        // CRITICAL: This is the updated destination URL for your Hugging Face Organization Space.
        // It now points to: https://AI4ALL3DCNN-neuroscan-backend.hf.space/:path*
        destination: 'https://AI4ALL3DCNN-neuroscan-backend.hf.space/:path*',
      },
    ];
  },
};

export default nextConfig;