// src/components/ProjectDemoDialog.tsx
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Badge } from './ui/badge';
import { Region } from '../types';
import { ExternalLink } from 'lucide-react';

interface ProjectDemoDialogProps {
  region: Region | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ProjectDemoDialog({ region, open, onOpenChange }: ProjectDemoDialogProps) {
  if (!region?.project) return null;

  const { title, blurb, href, keywords } = region.project;

  const videoUrl = '';
  const demoUrl = href;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl text-primary">{title}</DialogTitle>
          <DialogDescription>{blurb}</DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Keywords */}
          <div className="flex flex-wrap gap-2">
            {keywords.map((keyword) => (
              <Badge key={keyword} variant="outline">
                {keyword}
              </Badge>
            ))}
          </div>

          {/* Video Preview (if available) */}
          {videoUrl && (
            <div className="rounded-lg overflow-hidden border">
              <video 
                autoPlay 
                loop 
                muted 
                playsInline
                className="w-full"
              >
                <source src={videoUrl} type="video/mp4" />
              </video>
            </div>
          )}

          {/* Live Demo (iframe or link) */}
          {demoUrl && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold">Live Demo</h3>
              
              {/* Try to embed as iframe first */}
              <div className="rounded-lg overflow-hidden border bg-background h-[500px]">
                <iframe
                  src={demoUrl}
                  className="w-full h-full"
                  title={`${title} Demo`}
                  sandbox="allow-scripts allow-same-origin"
                />
              </div>

              {/* External link fallback */}
              <a
                href={demoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
              >
                Open in new window
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          )}

          {/* Project Description */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">About This Project</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {blurb}
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
