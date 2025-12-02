# Medical Images Setup Guide

## What Was Added

I've enhanced the medical-themed NeuroScan design with professional medical imagery. The images are displayed with **very low opacity (6-8%)** and strong overlays to maintain a professional, subtle aesthetic.

## Current Status: ✅ Working with Placeholders

The site is currently running with **SVG placeholder images** at:
- `/public/images/brain-scan-hero.svg` (Hero section background)
- `/public/images/medical-team.svg` (Clinical Evidence section background)

These placeholders will display subtle medical-themed patterns until you add real photographs.

## How to Add Real Medical Images

### Step 1: Download Royalty-Free Medical Images

Visit these **FREE, copyright-free** sources:

#### For Brain Scan (brain-scan-hero.jpg):
1. **Unsplash** (recommended):
   - https://unsplash.com/s/photos/brain-scan
   - https://unsplash.com/s/photos/mri-scan
   - Search: "brain mri", "cerebral angiography", "neurology scan"

2. **Pexels**:
   - https://www.pexels.com/search/brain%20scan/
   - Search: "brain scan medical", "MRI scan"

3. **Pixabay**:
   - https://pixabay.com/images/search/brain%20scan/

**Recommended images**: Blue/cool-toned MRI scans, angiography, or brain CT scans

#### For Medical Team (medical-team.jpg):
1. **Unsplash** (recommended):
   - https://unsplash.com/s/photos/doctor
   - https://unsplash.com/s/photos/radiologist
   - Search: "radiologist reviewing scans", "medical professionals", "doctor hospital"

2. **Pexels**:
   - https://www.pexels.com/search/doctor/
   - https://www.pexels.com/search/medical%20team/

3. **Pixabay**:
   - https://pixabay.com/images/search/doctor/

**Recommended images**: Medical professionals, doctors reviewing scans, healthcare team in clinical setting

### Step 2: Prepare the Images

1. Download the images you like
2. **Rename them exactly**:
   - `brain-scan-hero.jpg`
   - `medical-team.jpg`
3. **Optimize for web** (recommended):
   - Resolution: 1920x1080 or similar
   - Format: JPG
   - File size: Under 500KB each (use compression tools like TinyJPG.com)

### Step 3: Replace Placeholder Images

1. Navigate to: `/Users/chimin/Documents/Projects/Neuroscan-Frontend-Medical/public/images/`

2. **Delete or rename the placeholder SVG files**:
   ```bash
   cd /Users/chimin/Documents/Projects/Neuroscan-Frontend-Medical/public/images/
   rm brain-scan-hero.svg medical-team.svg
   ```

3. **Add your downloaded JPG images**:
   - Copy `brain-scan-hero.jpg` to this directory
   - Copy `medical-team.jpg` to this directory

4. **Update the code** to use JPG instead of SVG:
   - Edit `app/page.tsx`
   - Change `src="/images/brain-scan-hero.svg"` → `src="/images/brain-scan-hero.jpg"`
   - Change `src="/images/medical-team.svg"` → `src="/images/medical-team.jpg"`
   - Remove the `unoptimized` prop from both Image components

5. **Refresh your browser** - the real images should now appear!

## License Verification

All suggested sources (Unsplash, Pexels, Pixabay) offer:
- ✅ **Free for commercial use**
- ✅ **No attribution required** (though appreciated)
- ✅ **Can be modified**
- ✅ **Safe for production deployment**

Always verify the specific license for each image you download.

## Alternative: Use High-Quality Placeholder Services

If you want to test without downloading, you can temporarily use:
- https://placehold.co/1920x1080/e2e8f0/1e40af?text=Brain+Scan
- https://placehold.co/1920x1080/f1f5f9/1e40af?text=Medical+Team

Replace the image `src` paths with these URLs (add `unoptimized` prop for external URLs).

## Design Philosophy

The images are intentionally:
- **Very low opacity** (6-8%) - they provide texture, not distraction
- **Heavily overlayed** - white/blue gradients maintain professional aesthetic
- **Subtle** - the focus remains on the content and functionality
- **Professional** - medical-grade appearance suitable for clinical contexts

## Troubleshooting

**Images not showing?**
1. Check file names are **exactly**: `brain-scan-hero.jpg` and `medical-team.jpg`
2. Verify files are in `/public/images/` directory
3. Clear browser cache (Cmd/Ctrl + Shift + R)
4. Check browser console for 404 errors
5. Ensure you updated `.svg` to `.jpg` in the code

**Images too visible/distracting?**
- The opacity is set very low (0.06-0.08) in the code
- Don't change this - it's intentionally subtle for professional appearance

## Current Site

Visit: **http://localhost:3003** to see the medical-themed version with placeholder images.

Once you add real images, the site will have a more authentic medical/clinical aesthetic while maintaining the professional, minimal design.
