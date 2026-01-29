import { MetricCard } from "./MetricCard";
import {
  Package,
  Truck,
  Users,
  DollarSign,
  Leaf,
  Timer,
  Target,
  Zap,
} from "lucide-react";

export function QuickStats() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <MetricCard
        title="Active Deliveries"
        value="3,847"
        change={12.5}
        changeLabel="vs yesterday"
        icon={Package}
        iconColor="cyan"
        delay={0}
      />
      <MetricCard
        title="Online Drivers"
        value="847"
        change={8.2}
        changeLabel="vs yesterday"
        icon={Users}
        iconColor="success"
        delay={100}
      />
      <MetricCard
        title="Today's Revenue"
        value="AED 284K"
        change={15.8}
        changeLabel="vs yesterday"
        icon={DollarSign}
        iconColor="gold"
        delay={200}
      />
      <MetricCard
        title="Carbon Saved"
        value="1.2 tons"
        change={23.4}
        changeLabel="EV routes"
        icon={Leaf}
        iconColor="success"
        delay={300}
      />
    </div>
  );
}

export function PerformanceStats() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <MetricCard
        title="Avg. Delivery Time"
        value="24 min"
        change={-8.3}
        changeLabel="faster"
        icon={Timer}
        iconColor="cyan"
        delay={0}
      />
      <MetricCard
        title="Success Rate"
        value="98.7%"
        change={0.5}
        changeLabel="this week"
        icon={Target}
        iconColor="success"
        delay={100}
      />
      <MetricCard
        title="Fleet Utilization"
        value="87%"
        change={4.2}
        changeLabel="optimized"
        icon={Truck}
        iconColor="gold"
        delay={200}
      />
      <MetricCard
        title="System Sync"
        value="28ms"
        change={-12}
        changeLabel="latency"
        icon={Zap}
        iconColor="cyan"
        delay={300}
      />
    </div>
  );
}
