// src/config/projects.ts
// Maps region IDs → project info. Add entries here to assign projects to bricks.
// PROJECT_IDS is used by useFamilyLoader to keep project shapes always loaded.
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

/** Frozen set of region IDs that have project assignments — used by useFamilyLoader
 *  to guarantee these shapes are always in the DOM (always glowing). */
export const PROJECT_IDS: ReadonlySet<number> = Object.freeze(
  new Set(Object.keys(PROJECT_ASSIGNMENTS).map(Number)),
);
