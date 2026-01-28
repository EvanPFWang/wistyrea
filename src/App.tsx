// src/App.tsx
import { useState } from 'react';
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
  } = useMuralController();

  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [dialogOpen, setDialogOpen] = useState(false);

  const handleCanvasPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    handlePointerMove(e);
    setMousePosition({ x: e.clientX, y: e.clientY });
  };

  const handleCanvasClick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    handleClick(e);
    if (selectedRegion?.project) {
      setDialogOpen(true);
    }
  };

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
        canvasRef={canvasRef}
        onPointerMove={handleCanvasPointerMove}
        onClick={handleCanvasClick}
        getRegionColor={getRegionColor}
        fetchMask={fetchMask}
      />

      {/* Hover Preview Card */}
      {hoveredRegion?.project && (
        <ProjectPreviewCard
          region={hoveredRegion}
          position={mousePosition}
        />
      )}

      {/* Full Demo Dialog */}
      <ProjectDemoDialog
        region={selectedRegion}
        open={dialogOpen}
        onOpenChange={(open) => {
          setDialogOpen(open);
          if (!open) setSelectedRegion(null);
        }}
      />
    </div>
  );
}
