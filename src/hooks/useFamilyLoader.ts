// src/hooks/useFamilyLoader.ts
// Dynamic Morton-ordered family loader.
// Loads a window of ±HALF_FAMILY shape IDs around the hovered region,
// debounced to prevent thrashing during rapid cursor movement.

import { useState, useRef, useEffect, useCallback } from 'react';
import { Region } from '../types';

const HALF_FAMILY = 50;
const DEBOUNCE_MS = 50;

export interface FamilyState {
  /** Set of region IDs currently in the loaded family */
  familyIds: Set<number>;
  /** The anchor ID that generated this family (0 = no hover) */
  anchorId: number;
}

export function useFamilyLoader(
  hoveredRegion: Region | null,
  totalRegions: number,
): FamilyState {
  const [family, setFamily] = useState<FamilyState>({
    familyIds: new Set(),
    anchorId: 0,
  });

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastAnchorRef = useRef(0);

  const computeFamily = useCallback(
    (anchorId: number): Set<number> => {
      if (anchorId <= 0 || totalRegions <= 0) return new Set();
      const lo = Math.max(1, anchorId - HALF_FAMILY);
      const hi = Math.min(totalRegions, anchorId + HALF_FAMILY);
      const ids = new Set<number>();
      for (let i = lo; i <= hi; i++) ids.add(i);
      return ids;
    },
    [totalRegions],
  );

  useEffect(() => {
    const newAnchor = hoveredRegion?.id ?? 0;

    // Skip if anchor unchanged
    if (newAnchor === lastAnchorRef.current) return;
    lastAnchorRef.current = newAnchor;

    // Clear previous debounce
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }

    // No hover → clear family immediately (no debounce on exit)
    if (newAnchor <= 0) {
      setFamily({ familyIds: new Set(), anchorId: 0 });
      return;
    }

    // Debounce family computation to avoid thrashing during rapid scrub
    timerRef.current = setTimeout(() => {
      setFamily({
        familyIds: computeFamily(newAnchor),
        anchorId: newAnchor,
      });
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [hoveredRegion, computeFamily]);

  return family;
}
