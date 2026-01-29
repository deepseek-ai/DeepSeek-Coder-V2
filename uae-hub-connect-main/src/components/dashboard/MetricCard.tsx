import { cn } from "@/lib/utils";
import { LucideIcon, TrendingUp, TrendingDown } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  iconColor?: "gold" | "cyan" | "success" | "warning" | "destructive";
  className?: string;
  delay?: number;
}

export function MetricCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  iconColor = "gold",
  className,
  delay = 0,
}: MetricCardProps) {
  const isPositive = change && change > 0;
  const isNegative = change && change < 0;

  const iconColorClasses = {
    gold: "text-primary bg-primary/20",
    cyan: "text-cyan bg-cyan/20",
    success: "text-success bg-success/20",
    warning: "text-warning bg-warning/20",
    destructive: "text-destructive bg-destructive/20",
  };

  return (
    <div
      className={cn(
        "metric-card animate-fade-in-up",
        className
      )}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Icon */}
      <div className="flex items-start justify-between mb-4">
        <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center", iconColorClasses[iconColor])}>
          <Icon className="w-6 h-6" />
        </div>
        {change !== undefined && (
          <div className={cn("kpi-badge", isPositive && "positive", isNegative && "negative")}>
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span>{Math.abs(change)}%</span>
          </div>
        )}
      </div>

      {/* Value */}
      <div className="mb-1">
        <span className="font-display text-3xl font-bold text-gradient-gold">{value}</span>
      </div>

      {/* Title & Change Label */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">{title}</p>
        {changeLabel && (
          <p className="text-xs text-muted-foreground">{changeLabel}</p>
        )}
      </div>

      {/* Decorative Elements */}
      <div className="absolute bottom-0 right-0 w-32 h-32 opacity-5">
        <Icon className="w-full h-full" />
      </div>
    </div>
  );
}
