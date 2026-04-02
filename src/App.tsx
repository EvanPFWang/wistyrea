// src/App.tsx
import { useMemo } from 'react';
import { useMuralController } from './hooks/useMuralController';
import { MuralCanvas } from './components/MuralCanvas';
import { ProjectPreviewCard } from './components/ProjectPreviewCard';
import { ProjectDemoDialog } from './components/ProjectDemoDialog';

export function App() {
  const {
    metadata,
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
    activeRegions,
    bboxToViewport,
  } = useMuralController();

  const activeRegionsList = useMemo(() => activeRegions(), [activeRegions]);

  const handleCanvasPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    handlePointerMove(e);
  };

  const handleCanvasClick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    handleClick(e);
  };

  // Compute viewport bbox for hovered region (used by preview card positioning)
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
      </div>

      {/* Mural Canvas */}
      <MuralCanvas
        baseImage={baseImage}
        hoveredRegion={hoveredRegion}
        activeRegions={activeRegionsList}
        canvasRef={canvasRef}
        onPointerMove={handleCanvasPointerMove}
        onClick={handleCanvasClick}
        getRegionColor={getRegionColor}
        fetchMask={fetchMask}
      />

      {/* Quadrant-Based Preview Card */}
      {hoveredRegion?.project && hoveredViewportBbox && (
        <ProjectPreviewCard
          region={hoveredRegion}
          viewportBbox={hoveredViewportBbox}
        />
      )}

      {/* Full Demo Dialog (on click) */}
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
