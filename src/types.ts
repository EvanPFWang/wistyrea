// src/types.ts

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface ProjectInfo {
  title: string;
  href: string;
  blurb: string;
}

export interface Region {
  id: number;
  bbox: BoundingBox;
  centroid: Point;
  mask: string;
  project?: ProjectInfo;
}

export interface Metadata {
  version: string;
  dimensions: {
    width: number;
    height: number;
  };
  total_regions: number;
  background_id: number;
  regions: Region[];
}

export interface PaletteColor {
  r: number;
  g: number;
  b: number;
}

export interface PaletteData {
  background_id: number;
  map: Record<string, PaletteColor>;
}

export interface ProjectType {
  prefix: string;
  titles: string[];
}

export interface ControllerConfig {
  dataPath?: string;
  enableDebug?: boolean;
  maxFPS?: number;
}