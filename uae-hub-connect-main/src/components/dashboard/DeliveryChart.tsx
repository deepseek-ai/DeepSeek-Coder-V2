import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { cn } from "@/lib/utils";

const generateHourlyData = () => {
  const hours = [];
  const now = new Date();
  for (let i = 23; i >= 0; i--) {
    const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
    const baseValue = 100 + Math.sin((24 - i) / 3) * 80;
    hours.push({
      time: hour.toLocaleTimeString("en-AE", { hour: "2-digit", minute: "2-digit", hour12: false }),
      deliveries: Math.floor(baseValue + Math.random() * 40),
      efficiency: Math.floor(94 + Math.random() * 6),
    });
  }
  return hours;
};

export function DeliveryChart() {
  const data = useMemo(() => generateHourlyData(), []);

  return (
    <div className="glass-card p-6 animate-fade-in-up" style={{ animationDelay: "600ms" }}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-display text-lg font-semibold text-gradient-gold">Delivery Volume</h3>
          <p className="text-sm text-muted-foreground">24-hour performance overview</p>
        </div>
        <div className="flex gap-2">
          {["1H", "6H", "24H", "7D"].map((period, idx) => (
            <button
              key={period}
              className={cn(
                "px-3 py-1 rounded-lg text-xs font-medium transition-colors",
                idx === 2
                  ? "bg-primary/20 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
            >
              {period}
            </button>
          ))}
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="deliveryGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(185 100% 50%)" stopOpacity={0.4} />
                <stop offset="100%" stopColor="hsl(185 100% 50%)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="efficiencyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(43 50% 58%)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(43 50% 58%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(220 30% 18%)"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(215 20% 55%)", fontSize: 10 }}
              interval={3}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(215 20% 55%)", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(220 35% 10%)",
                borderColor: "hsl(220 30% 18%)",
                borderRadius: "12px",
                boxShadow: "0 8px 32px hsl(220 40% 2% / 0.5)",
              }}
              labelStyle={{ color: "hsl(43 50% 58%)", fontFamily: "Orbitron" }}
              itemStyle={{ color: "hsl(210 40% 96%)" }}
            />
            <Area
              type="monotone"
              dataKey="deliveries"
              stroke="hsl(185 100% 50%)"
              strokeWidth={2}
              fill="url(#deliveryGradient)"
              animationDuration={2000}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-border/50">
        <div className="text-center">
          <p className="font-display text-xl font-bold text-cyan">3,847</p>
          <p className="text-xs text-muted-foreground">Total Today</p>
        </div>
        <div className="text-center">
          <p className="font-display text-xl font-bold text-primary">174</p>
          <p className="text-xs text-muted-foreground">Peak Hour</p>
        </div>
        <div className="text-center">
          <p className="font-display text-xl font-bold text-success">+12%</p>
          <p className="text-xs text-muted-foreground">vs Last Week</p>
        </div>
      </div>
    </div>
  );
}
