// src/config/projects.ts
// Maps region IDs to project info. Add entries here to assign projects to bricks.
import { ProjectInfo } from '../types';

export const PROJECT_ASSIGNMENTS: Record<number, ProjectInfo> = {
  21: {
    title: 'Black-Scholes Option Pricer',
    href: 'https://wistyrea.com/black-scholes',
    blurb:
      'Interactive option pricing with parareal iterators. Visualizes convergence of parallel-in-time solvers for the Black-Scholes PDE.',
    keywords: ['Finance', 'PDE', 'Parareal', 'Streamlit'],
  },
  22: {
    title: 'Yeast Reverse Image Segmentation',
    href: 'https://wistyrea.com/yeast-segmentation',
    blurb:
      'Synthetic yeast cell image generation from segmentation labels. Ellipse-based morphology modeling with fluorescence rendering, Perlin noise boundary perturbation, and vectorized operations for 8x speedup.',
    keywords: ['OpenCV', 'Segmentation', 'Fluorescence', 'Microscopy', 'Perlin Noise'],
  },
};
