import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Truck, Package, User, AlertTriangle, CheckCircle, Clock } from "lucide-react";

interface Activity {
  id: string;
  type: "delivery" | "driver" | "vendor" | "alert" | "success";
  message: string;
  timestamp: Date;
  entity?: string;
}

const mockActivities: Activity[] = [
  { id: "1", type: "success", message: "Order #ORD-7842 delivered successfully", timestamp: new Date(), entity: "Noon" },
  { id: "2", type: "driver", message: "Driver Ahmed K. went online in Dubai Marina", timestamp: new Date(Date.now() - 60000) },
  { id: "3", type: "delivery", message: "New order assigned to Vendor DSL-003", timestamp: new Date(Date.now() - 120000), entity: "Amazon" },
  { id: "4", type: "alert", message: "Traffic delay detected on Sheikh Zayed Rd", timestamp: new Date(Date.now() - 180000) },
  { id: "5", type: "vendor", message: "Careem Fleet: 12 new vehicles online", timestamp: new Date(Date.now() - 240000), entity: "Careem" },
  { id: "6", type: "success", message: "Batch #B-1204 completed - 24 deliveries", timestamp: new Date(Date.now() - 300000) },
  { id: "7", type: "driver", message: "Driver Fatima S. completed 50th delivery today", timestamp: new Date(Date.now() - 360000) },
  { id: "8", type: "delivery", message: "Express delivery en route to Business Bay", timestamp: new Date(Date.now() - 420000), entity: "Keeta" },
];

const activityIcons = {
  delivery: Package,
  driver: User,
  vendor: Truck,
  alert: AlertTriangle,
  success: CheckCircle,
};

const activityColors = {
  delivery: "text-info bg-info/20",
  driver: "text-primary bg-primary/20",
  vendor: "text-cyan bg-cyan/20",
  alert: "text-warning bg-warning/20",
  success: "text-success bg-success/20",
};

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

export function LiveActivityFeed() {
  const [activities, setActivities] = useState<Activity[]>(mockActivities);

  // Simulate new activities coming in
  useEffect(() => {
    const interval = setInterval(() => {
      const newActivity: Activity = {
        id: Date.now().toString(),
        type: ["delivery", "driver", "vendor", "success"][Math.floor(Math.random() * 4)] as Activity["type"],
        message: [
          "New order received from Dubai Mall area",
          "Driver completed route optimization",
          "Vendor fleet availability updated",
          "Delivery confirmed at destination",
        ][Math.floor(Math.random() * 4)],
        timestamp: new Date(),
        entity: ["Noon", "Amazon", "Careem", "Keeta"][Math.floor(Math.random() * 4)],
      };
      setActivities((prev) => [newActivity, ...prev.slice(0, 7)]);
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-card p-6 h-full animate-fade-in-up" style={{ animationDelay: "400ms" }}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-display text-lg font-semibold text-gradient-gold">Live Activity Feed</h3>
          <p className="text-sm text-muted-foreground">Real-time ecosystem updates</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-success/20 text-success text-xs">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          Streaming
        </div>
      </div>

      <div className="space-y-3 max-h-[380px] overflow-y-auto pr-2">
        {activities.map((activity, index) => {
          const Icon = activityIcons[activity.type];
          return (
            <div
              key={activity.id}
              className={cn(
                "flex items-start gap-3 p-3 rounded-lg bg-secondary/30 border border-border/30 hover:border-primary/30 transition-all cursor-pointer group",
                index === 0 && "animate-scale-in ring-1 ring-primary/30"
              )}
            >
              <div
                className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
                  activityColors[activity.type]
                )}
              >
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-foreground leading-snug">{activity.message}</p>
                <div className="flex items-center gap-2 mt-1">
                  {activity.entity && (
                    <span className="px-2 py-0.5 rounded-full bg-secondary text-xs text-muted-foreground">
                      {activity.entity}
                    </span>
                  )}
                  <span className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {formatTimeAgo(activity.timestamp)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* View All Button */}
      <button className="w-full mt-4 py-2 rounded-lg border border-border/50 text-sm text-muted-foreground hover:text-foreground hover:border-primary/50 transition-all">
        View All Activity →
      </button>
    </div>
  );
}
