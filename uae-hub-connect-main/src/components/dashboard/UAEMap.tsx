import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface DeliveryMarker {
  id: string;
  x: number;
  y: number;
  type: "active" | "pending" | "completed";
  emirate: string;
}

const mockMarkers: DeliveryMarker[] = [
  { id: "1", x: 75, y: 45, type: "active", emirate: "Dubai" },
  { id: "2", x: 72, y: 42, type: "active", emirate: "Dubai" },
  { id: "3", x: 78, y: 48, type: "pending", emirate: "Dubai" },
  { id: "4", x: 65, y: 40, type: "active", emirate: "Abu Dhabi" },
  { id: "5", x: 62, y: 38, type: "completed", emirate: "Abu Dhabi" },
  { id: "6", x: 82, y: 35, type: "active", emirate: "Sharjah" },
  { id: "7", x: 80, y: 32, type: "pending", emirate: "Sharjah" },
  { id: "8", x: 85, y: 30, type: "active", emirate: "Ajman" },
  { id: "9", x: 88, y: 28, type: "active", emirate: "Ras Al Khaimah" },
  { id: "10", x: 90, y: 35, type: "pending", emirate: "Fujairah" },
  { id: "11", x: 73, y: 50, type: "active", emirate: "Dubai" },
  { id: "12", x: 76, y: 43, type: "active", emirate: "Dubai" },
];

export function UAEMap() {
  const [markers, setMarkers] = useState<DeliveryMarker[]>(mockMarkers);
  const [activeEmirateStats, setActiveEmirateStats] = useState({
    Dubai: 142,
    "Abu Dhabi": 89,
    Sharjah: 67,
    Ajman: 23,
    "Ras Al Khaimah": 18,
    Fujairah: 12,
    "Umm Al Quwain": 5,
  });

  // Simulate real-time marker movements
  useEffect(() => {
    const interval = setInterval(() => {
      setMarkers((prev) =>
        prev.map((marker) => ({
          ...marker,
          x: marker.x + (Math.random() - 0.5) * 0.3,
          y: marker.y + (Math.random() - 0.5) * 0.3,
        }))
      );
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const markerColors = {
    active: "bg-cyan text-cyan",
    pending: "bg-warning text-warning",
    completed: "bg-success text-success",
  };

  return (
    <div className="glass-card p-6 h-full animate-fade-in-up" style={{ animationDelay: "200ms" }}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-display text-lg font-semibold text-gradient-gold">UAE Operations Map</h3>
          <p className="text-sm text-muted-foreground">Real-time delivery tracking across all emirates</p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
            <span className="text-xs text-muted-foreground">Active</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-warning" />
            <span className="text-xs text-muted-foreground">Pending</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-success" />
            <span className="text-xs text-muted-foreground">Completed</span>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="relative h-64 md:h-80 rounded-xl bg-secondary/30 overflow-hidden border border-border/50">
        {/* Grid Pattern Background */}
        <div className="absolute inset-0 grid-pattern opacity-30" />

        {/* UAE Shape Outline (Simplified) */}
        <svg viewBox="0 0 100 60" className="absolute inset-0 w-full h-full">
          <defs>
            <linearGradient id="mapGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.3" />
              <stop offset="100%" stopColor="hsl(var(--cyan))" stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <path
            d="M50 55 L55 50 L60 52 L68 48 L75 50 L82 45 L88 42 L92 38 L95 32 L92 28 L88 25 L82 22 L78 25 L72 28 L68 32 L62 35 L58 38 L52 40 L48 42 L42 45 L38 48 L32 50 L28 52 L22 50 L18 48 L15 45 L18 42 L22 40 L28 38 L35 40 L42 42 L48 45 L50 50 Z"
            fill="url(#mapGradient)"
            stroke="hsl(var(--primary))"
            strokeWidth="0.5"
            strokeOpacity="0.5"
          />
        </svg>

        {/* Delivery Markers */}
        {markers.map((marker, index) => (
          <div
            key={marker.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-1000"
            style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
          >
            <div
              className={cn(
                "map-marker",
                markerColors[marker.type]
              )}
              style={{ animationDelay: `${index * 200}ms` }}
            />
            <div
              className={cn(
                "absolute w-8 h-8 rounded-full opacity-20 animate-ping",
                marker.type === "active" && "bg-cyan",
                marker.type === "pending" && "bg-warning",
                marker.type === "completed" && "bg-success"
              )}
              style={{ animationDuration: "2s" }}
            />
          </div>
        ))}

        {/* Connection Lines (Animated) */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {markers.slice(0, 5).map((marker, i) => {
            const nextMarker = markers[(i + 1) % markers.length];
            return (
              <line
                key={`line-${i}`}
                x1={`${marker.x}%`}
                y1={`${marker.y}%`}
                x2={`${nextMarker.x}%`}
                y2={`${nextMarker.y}%`}
                stroke="hsl(var(--cyan))"
                strokeWidth="0.5"
                strokeOpacity="0.3"
                strokeDasharray="4 4"
              >
                <animate
                  attributeName="stroke-dashoffset"
                  values="8;0"
                  dur="1s"
                  repeatCount="indefinite"
                />
              </line>
            );
          })}
        </svg>

        {/* Emirate Labels */}
        <div className="absolute top-4 left-4 space-y-2">
          {Object.entries(activeEmirateStats).slice(0, 4).map(([emirate, count]) => (
            <div
              key={emirate}
              className="flex items-center gap-2 px-2 py-1 rounded-lg bg-card/80 backdrop-blur-sm text-xs"
            >
              <span className="text-muted-foreground">{emirate}</span>
              <span className="font-display font-bold text-primary">{count}</span>
            </div>
          ))}
        </div>

        {/* Live Counter */}
        <div className="absolute bottom-4 right-4 glass-card p-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
            <span className="font-display text-lg font-bold text-gradient-cyan">356</span>
            <span className="text-xs text-muted-foreground">Live Deliveries</span>
          </div>
        </div>
      </div>
    </div>
  );
}
