import { cn } from "@/lib/utils";
import { Building2, TrendingUp, Package, Clock } from "lucide-react";

interface Brand {
  id: string;
  name: string;
  logo: string;
  ordersToday: number;
  efficiency: number;
  avgDeliveryTime: string;
  status: "active" | "syncing" | "offline";
}

const mockBrands: Brand[] = [
  { id: "1", name: "Noon", logo: "🟡", ordersToday: 1247, efficiency: 98.5, avgDeliveryTime: "24min", status: "active" },
  { id: "2", name: "Amazon", logo: "🟠", ordersToday: 892, efficiency: 97.2, avgDeliveryTime: "28min", status: "active" },
  { id: "3", name: "Careem", logo: "🟢", ordersToday: 654, efficiency: 96.8, avgDeliveryTime: "18min", status: "active" },
  { id: "4", name: "Keeta", logo: "🔵", ordersToday: 423, efficiency: 99.1, avgDeliveryTime: "15min", status: "syncing" },
  { id: "5", name: "Porter", logo: "🟣", ordersToday: 287, efficiency: 95.4, avgDeliveryTime: "32min", status: "active" },
];

export function BrandOverview() {
  return (
    <div className="glass-card p-6 animate-fade-in-up" style={{ animationDelay: "300ms" }}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
            <Building2 className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-gradient-gold">Connected Brands</h3>
            <p className="text-sm text-muted-foreground">5 active integrations</p>
          </div>
        </div>
        <button className="px-4 py-2 rounded-lg bg-primary/10 text-primary text-sm font-medium hover:bg-primary/20 transition-colors">
          + Add Brand
        </button>
      </div>

      <div className="space-y-3">
        {mockBrands.map((brand, index) => (
          <div
            key={brand.id}
            className={cn(
              "flex items-center gap-4 p-4 rounded-xl bg-secondary/30 border border-border/30 hover:border-primary/30 transition-all cursor-pointer group",
              "animate-slide-in-left"
            )}
            style={{ animationDelay: `${300 + index * 100}ms` }}
          >
            {/* Logo */}
            <div className="w-12 h-12 rounded-xl bg-card flex items-center justify-center text-2xl">
              {brand.logo}
            </div>

            {/* Brand Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="font-semibold text-foreground">{brand.name}</h4>
                <div
                  className={cn(
                    "w-2 h-2 rounded-full",
                    brand.status === "active" && "bg-success",
                    brand.status === "syncing" && "bg-warning animate-pulse",
                    brand.status === "offline" && "bg-destructive"
                  )}
                />
              </div>
              <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Package className="w-3 h-3" />
                  {brand.ordersToday.toLocaleString()} orders
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {brand.avgDeliveryTime}
                </span>
              </div>
            </div>

            {/* Efficiency Score */}
            <div className="text-right">
              <div className="flex items-center gap-1 justify-end">
                <span className="font-display text-xl font-bold text-success">{brand.efficiency}%</span>
                <TrendingUp className="w-4 h-4 text-success" />
              </div>
              <p className="text-xs text-muted-foreground">Efficiency</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
