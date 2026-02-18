# MCMC Sampler Analysis - Navigation Guide

This directory contains a comprehensive analysis of MCMC sampler performance on two benchmark distributions (Rosenbrock and Neal's Funnel).

---

## 📚 DOCUMENTATION INDEX

### Start Here:

1. **Executive_Summary.md** ⭐ START HERE
   - Quick overview of all findings
   - Performance tables and key numbers
   - Decision tree for choosing samplers
   - 5-10 minute read

2. **Visual_Summary.md**
   - Highlights from visual analysis
   - Key plot interpretations
   - Common diagnostic patterns
   - Reference guide for plot files
   - 10-15 minute read

3. **MCMC_Analysis_Report.md**
   - Complete detailed analysis
   - All 22 plots analyzed systematically
   - Section-by-section breakdowns
   - Technical deep-dive
   - 30-45 minute read

### Source Files:

4. **assignment-01-starter.html**
   - Original Jupyter notebook export
   - Contains all plots and code
   - 2.2 MB file with embedded images

5. **extracted_plots/**
   - All 22 plots extracted as PNG files
   - Organized by number and description
   - Ready for presentations/papers

---

## 🎯 QUICK NAVIGATION BY TOPIC

### Looking for performance numbers?
→ See **Executive_Summary.md** sections:
- "Performance at a Glance"
- "By the Numbers"

### Want to understand the plots?
→ See **Visual_Summary.md** sections:
- "Scatter Plot Comparison"
- "Trace Plot Red Flags"
- "Plot File Reference"

### Need technical details?
→ See **MCMC_Analysis_Report.md** sections:
- Section 1: Rosenbrock analysis
- Section 2: Neal's Funnel analysis
- Section 6: Critical Issues & Recommendations

### Preparing a presentation?
→ Use **extracted_plots/** folder
- Files are numbered and labeled
- Key plots: 01, 10, 16, 19
- See Visual_Summary.md for plot descriptions

---

## 🔍 FIND BY SAMPLER

### RWMH (Random Walk Metropolis-Hastings)
- **Executive Summary**: "RWMH" sections
- **Visual Summary**: "Autocorrelation Paradox"
- **Full Report**: Sections 1.2, 2.2
- **Plots**: 01, 02, 06, 07

### HMC (Hamiltonian Monte Carlo)
- **Executive Summary**: "HMC" sections
- **Visual Summary**: "Trace Plot Red Flags"
- **Full Report**: Sections 1.3, 2.3
- **Plots**: 01, 03, 06, 08

### NUTS (No-U-Turn Sampler)
- **Executive Summary**: "NUTS: The Reliable Workhorse"
- **Visual Summary**: "Scatter Plot Comparison"
- **Full Report**: Sections 1.4, 2.4
- **Plots**: 10, 11, 12, 13, 14, 15

### GG-NUTS (Geometry-Gated NUTS)
- **Executive Summary**: "Critical Issues" ⚠️
- **Visual Summary**: "Critical Finding" 🚨
- **Full Report**: Sections 1.5, 2.5, 3, 6
- **Plots**: 16, 17, 18, 19, 20, 21

---

## 📊 FIND BY DISTRIBUTION

### Rosenbrock (Banana)
- **Key Finding**: GG-NUTS catastrophic failure (ESS=24)
- **Plots**: 00-05, 10-12, 16-18
- **Analysis**: Report Sections 1.1-1.5

### Neal's Funnel
- **Key Finding**: GG-NUTS best performance (ESS=864)
- **Plots**: 05-09, 13-15, 19-21
- **Analysis**: Report Sections 2.1-2.5

---

## 🎓 FIND BY DIAGNOSTIC TYPE

### Scatter Plots (Coverage)
- **Files**: 01, 06, 10, 13, 16, 19
- **Analysis**: Visual Summary "Scatter Plot Comparison"
- **What to look for**: Full distribution coverage

### Trace Plots (Mixing)
- **Files**: 02, 03, 07, 08, 11, 14, 17, 20
- **Analysis**: Visual Summary "Trace Plot Red Flags"
- **What to look for**: Rapid oscillation, no plateaus

### Autocorrelation (Independence)
- **Files**: 04, 09, 12, 15, 18, 21
- **Analysis**: Visual Summary "Autocorrelation Paradox"
- **What to look for**: Fast decay to zero

### Marginal Distributions
- **Files**: 02, 03, 07, 08, 11, 14, 17, 20
- **Analysis**: Report Section 7.3
- **What to look for**: Match to target distribution

---

## 🚨 CRITICAL FINDINGS

### 1. GG-NUTS Rosenbrock Failure
- **Where**: Executive Summary, Issue #1
- **What**: ESS drops from 304 (NUTS) to 24 (GG-NUTS)
- **Why**: Geometry detection misclassifies curved ridge
- **Evidence**: Plots 16-18, ablation study results
- **Impact**: Method unusable on curved manifolds

### 2. Fixed-Mixture Reveals Core Problem
- **Where**: Report Section 3, Visual Summary "Ablation Study"
- **What**: Random kernel selection outperforms geometry-based
- **Why**: Proves detection mechanism is fundamentally flawed
- **Numbers**: Fixed-Mix ESS=223 vs GG-NUTS ESS=24 on Rosenbrock

### 3. Autocorrelation Can Mislead
- **Where**: Visual Summary "Autocorrelation Paradox"
- **What**: RWMH has excellent autocorrelation but limited coverage
- **Why**: Low correlation ≠ good sampling
- **Lesson**: Always check multiple diagnostics together

---

## 📈 PERFORMANCE SUMMARY TABLES

### Best ESS by Distribution:
```
Rosenbrock:
  RWMH:      23,972 (but slow, no gradients)
  NUTS:         304 ← Best gradient-based ✓
  Fixed-Mix:    223
  GG-NUTS:       24 🚨

Funnel:
  RWMH:      16,230 (limited exploration)
  HMC:        1,929
  GG-NUTS:      864 ← Best overall ✓
  Fixed-Mix:    853
  NUTS:         637
```

### Best Efficiency (ESS/Grad):
```
Rosenbrock:
  NUTS:      0.00097 ✓
  Fixed-Mix: 0.00100
  HMC:       0.00014
  GG-NUTS:   0.00014 🚨

Funnel:
  GG-NUTS:   0.01226 ✓
  NUTS:      0.01233
  Fixed-Mix: 0.00957
  HMC:       0.00193
```

---

## 🔧 RECOMMENDATIONS QUICK REFERENCE

### For General Use:
✓ **Use NUTS** - reliable, consistent, well-tested

### For Hierarchical Models with Varying Scales:
⚠️ **Consider GG-NUTS** - but test thoroughly first!
- Validate ESS > NUTS on your specific problem
- Have fallback to vanilla NUTS ready
- Monitor for failure modes

### For Curved Manifolds (like Rosenbrock):
🚨 **Avoid GG-NUTS** - use NUTS instead

### For GG-NUTS Development:
🔧 **Critical fixes needed:**
1. Revise geometry detection metric
2. Add runtime safeguards (ESS monitoring)
3. Fix threshold selection mechanism
4. Expand test suite (≥10 distributions)

See Report Section 6 for detailed recommendations.

---

## 📁 FILE STRUCTURE

```
assignment-1-Achuthan2909/
│
├── README.md                      ← You are here
├── Executive_Summary.md           ← Start here (5-10 min)
├── Visual_Summary.md              ← Plot guide (10-15 min)
├── MCMC_Analysis_Report.md        ← Full analysis (30-45 min)
│
├── assignment-01-starter.html     ← Original notebook
│
└── extracted_plots/               ← All plots as PNG
    ├── 00_plot_0.png              ← Rosenbrock density
    ├── 01_random_walk_mh.png      ← RWMH/HMC scatter (Rosenbrock)
    ├── 02-05_*_rosenbrock.png     ← RWMH/HMC diagnostics
    ├── 06-09_*_funnel.png         ← RWMH/HMC on Funnel
    ├── 10-12_nuts_rosenbrock.png  ← NUTS on Rosenbrock
    ├── 13-15_nuts_funnel.png      ← NUTS on Funnel
    ├── 16-18_gg_nuts_rosenbrock*  ← GG-NUTS Rosenbrock ⚠️
    └── 19-21_gg_nuts_funnel*      ← GG-NUTS Funnel ✓
```

---

## 💡 HOW TO USE THIS ANALYSIS

### For Assignment Review:
1. Read Executive_Summary.md
2. Look at key plots (01, 10, 16, 19)
3. Review "Critical Issues" section

### For Understanding MCMC Diagnostics:
1. Read Visual_Summary.md
2. Study plot examples with explanations
3. Use as reference for your own analyses

### For Improving GG-NUTS:
1. Read Report Section 6 (Recommendations)
2. Study ablation results (Section 3)
3. Examine failure mode (Section 1.5)

### For Paper/Presentation:
1. Use performance tables from Executive_Summary
2. Select key plots from extracted_plots/
3. Reference specific findings with section numbers

---

## 🎓 KEY LESSONS

1. **Trust numerical diagnostics over visuals**
   - GG-NUTS plots look fine but ESS=24 reveals truth

2. **Ablation studies are essential**
   - Fixed-mixture baseline revealed core problem

3. **One size doesn't fit all**
   - GG-NUTS is specialist tool, not general-purpose

4. **Always have a fallback**
   - Novel methods can fail catastrophically

5. **Multiple diagnostics required**
   - Scatter + trace + autocorrelation + ESS

---

## 📞 QUESTIONS?

### "Which document should I read first?"
→ Executive_Summary.md for quick overview

### "I want to understand a specific plot"
→ Visual_Summary.md → "Plot File Reference" section

### "What's wrong with GG-NUTS?"
→ Executive_Summary.md → "Critical Issues" section
→ Full details in Report Section 6

### "How do I choose a sampler?"
→ Executive_Summary.md → "Decision Tree" section

### "What needs to be fixed?"
→ Report Section 6.2 → "Recommendations for GG-NUTS Improvement"

### "Can I use GG-NUTS for my hierarchical model?"
→ Maybe! See Executive_Summary.md → "For Hierarchical Models"
→ Test thoroughly and have NUTS as fallback

---

## 📊 ANALYSIS STATS

- **Total plots analyzed**: 22
- **Samplers compared**: 4 (plus 1 ablation)
- **Distributions tested**: 2
- **Critical issues found**: 1 major
- **Pages of analysis**: ~50+ across all documents
- **Analysis date**: February 15, 2026

---

## ⚡ TL;DR

**3-sentence summary:**
1. NUTS is the reliable gold standard - use as default.
2. GG-NUTS excels on Neal's Funnel (ESS=864) but catastrophically fails on Rosenbrock (ESS=24).
3. The geometry detection mechanism needs fundamental revision before GG-NUTS can be recommended for general use.

**Bottom line:** Use NUTS. Only consider GG-NUTS for hierarchical models after thorough validation.

---

**Happy reading! Start with Executive_Summary.md ⭐**
