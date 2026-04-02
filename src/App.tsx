// src/App.tsx
import { useMemo } from 'react';
import { useMuralController } from './hooks/useMuralController';
import { useFamilyLoader } from './hooks/useFamilyLoader';
import { MuralCanvas } from './components/MuralCanvas';
import { ProjectPreviewCard } from './components/ProjectPreviewCard';
import { ProjectDemoDialog } from './components/ProjectDemoDialog';

export function App() {
  const {
    metadata,
    regions,
    hoveredRegion,
    selectedRegion,
    baseImage,
    loading,
    canvasRef,
    handlePointerMove,
    handleClick,
    fetchMask,
    getRegionColor,
    setSelectedRegion,
    bboxToViewport,
  } = useMuralController();

  // Morton-ordered family loader: ±50 IDs around hovered region
  const { familyIds } = useFamilyLoader(
    hoveredRegion,
    metadata?.total_regions ?? 0,
  );

  // Viewport bbox for quadrant-based preview positioning
  const hoveredViewportBbox = useMemo(() => {
    if (!hoveredRegion?.project) return null;
    return bboxToViewport(hoveredRegion.bbox);
  }, [hoveredRegion, bboxToViewport]);

  if (loading) {
    return (
      <div className="w-screen h-screen flex items-center justify-center">
        <div className="text-primary text-lg animate-pulse-gold">
          Loading crown mural regions...
        </div>
      </div>
    );
  }

  return (
    <div className="w-screen h-screen relative overflow-hidden">
      {/* Stats Panel */}
      <div className="absolute top-3 right-3 bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3 text-xs font-mono space-y-1 z-40">
        <div>Regions: <span className="text-primary">{metadata?.total_regions ?? 0}</span></div>
        <div>Current: <span className="text-primary">
          {hoveredRegion ? `#${hoveredRegion.id}` : '-'}
        </span></div>
        <div>Family: <span className="text-primary">{familyIds.size}</span></div>
      </div>

      {/* Mural Canvas + Shape Layer */}
      <MuralCanvas
        baseImage={baseImage}
        hoveredRegion={hoveredRegion}
        familyIds={familyIds}
        regions={regions}
        canvasRef={canvasRef}
        onPointerMove={handlePointerMove}
        onClick={handleClick}
        getRegionColor={getRegionColor}
        fetchMask={fetchMask}
      />

      {/* Quadrant-Based Preview Card (hover on project shape) */}
      {hoveredRegion?.project && hoveredViewportBbox && (
        <ProjectPreviewCard
          region={hoveredRegion}
          viewportBbox={hoveredViewportBbox}
        />
      )}

      {/* Full Demo Dialog (click on project shape) */}
      <ProjectDemoDialog
        region={selectedRegion}
        open={!!selectedRegion?.project}
        onOpenChange={(open) => {
          if (!open) setSelectedRegion(null);
        }}
      />
    </div>
  );
}
