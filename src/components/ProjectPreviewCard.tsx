// src/components/ProjectPreviewCard.tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Region } from '../types';

interface ProjectPreviewCardProps {
  region: Region;
  position: { x: number; y: number };
}

export function ProjectPreviewCard({ region, position }: ProjectPreviewCardProps) {
  if (!region.project) return null;

  const { title, blurb } = region.project;

  // Example keywords - you can add these to your types.ts
  const keywords = ['WebGPU', 'Interactive', 'React']; // TODO: Add to project data

  return (
    <Card 
      className="fixed z-50 w-72 pointer-events-none animate-in fade-in-0 zoom-in-95 duration-200"
      style={{ 
        left: `${position.x + 16}px`, 
        top: `${position.y - 8}px`,
      }}
    >
      <CardHeader className="pb-3">
        <CardTitle className="text-base text-primary">{title}</CardTitle>
        <CardDescription className="text-xs">{blurb}</CardDescription>
      </CardHeader>
      <CardContent className="pb-3">
        <div className="flex flex-wrap gap-1.5">
          {keywords.map((keyword) => (
            <Badge key={keyword} variant="secondary" className="text-xs">
              {keyword}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
