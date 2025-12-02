# Medical Images for NeuroScan

This directory should contain the following royalty-free medical images:

## Required Images

### 1. `brain-scan-hero.jpg`
**Purpose**: Hero section background
**Specifications**:
- Resolution: 1920x1080 or higher
- Format: JPG
- Content: Brain MRI scan, CT scan, or angiography image (subtle, will be heavily overlayed)

**Suggested Sources**:
- **Unsplash**: Search "brain scan" or "MRI"
  - https://unsplash.com/s/photos/brain-scan
  - https://unsplash.com/s/photos/mri
- **Pexels**: Search "brain scan medical"
  - https://www.pexels.com/search/brain%20scan/
- **Pixabay**: Medical imaging category
  - https://pixabay.com/images/search/brain%20scan/

**Example searches**:
- "brain mri scan"
- "cerebral angiography"
- "medical brain imaging"
- "neurology scan"

---

### 2. `medical-team.jpg`
**Purpose**: Clinical Evidence section background
**Specifications**:
- Resolution: 1920x1080 or higher
- Format: JPG
- Content: Medical professionals, radiologists, doctors reviewing scans, or medical equipment

**Suggested Sources**:
- **Unsplash**:
  - https://unsplash.com/s/photos/doctor
  - https://unsplash.com/s/photos/medical-team
  - https://unsplash.com/s/photos/radiologist
- **Pexels**:
  - https://www.pexels.com/search/doctor/
  - https://www.pexels.com/search/medical%20team/
- **Pixabay**:
  - https://pixabay.com/images/search/doctor/

**Example searches**:
- "radiologist reviewing scans"
- "medical professionals"
- "doctor hospital"
- "healthcare team"

---

## License Information

All sources listed above offer **free, royalty-free images** under licenses that allow:
- Commercial use
- Modification
- No attribution required (though appreciated)

Always verify the specific license for each image you download.

## Alternative: Create Placeholder Images

If you want to test the layout without downloading images, you can use placeholder services:

1. **Via URL** (temporary solution):
   - Replace image paths with: `https://placehold.co/1920x1080/e2e8f0/1e40af?text=Brain+Scan`

2. **Solid Color Placeholders**:
   - Create simple JPG files with solid colors or gradients using any image editor

## Image Optimization

After adding images, consider optimizing them:
```bash
# Install sharp for Next.js image optimization (if not already installed)
npm install sharp

# Next.js will automatically optimize images at build/runtime
```

## Notes

- Images are displayed with **very low opacity** (6-8%) and **strong overlays**
- They serve as subtle texture/depth, not primary visual elements
- Professional medical aesthetic is maintained through restraint
- File sizes should be optimized (under 500KB each after compression)
