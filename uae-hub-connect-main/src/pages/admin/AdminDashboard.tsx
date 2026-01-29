import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import {
  Globe,
  Building2,
  Truck,
  Users,
  Shield,
  BarChart3,
  Settings,
  Bell,
  LogOut,
  Search,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Leaf,
  Activity,
  Eye,
  FileText,
  Zap,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface SystemAlert {
  id: string;
  type: "critical" | "warning" | "info";
  message: string;
  entity: string;
  time: string;
}

interface EntitySummary {
  type: string;
  total: number;
  active: number;
  pending: number;
  issues: number;
}

const systemAlerts: SystemAlert[] = [
  { id: "1", type: "critical", message: "Driver license expired", entity: "Mohammed K. (D-847)", time: "5 min ago" },
  { id: "2", type: "warning", message: "Vehicle inspection due", entity: "Fleet #23 (SwiftDel)", time: "1 hour ago" },
  { id: "3", type: "info", message: "New vendor application", entity: "RapidFleet LLC", time: "2 hours ago" },
  { id: "4", type: "warning", message: "Low driver availability", entity: "Abu Dhabi Zone", time: "3 hours ago" },
];

const entitySummaries: EntitySummary[] = [
  { type: "Brands", total: 5, active: 5, pending: 2, issues: 0 },
  { type: "Vendors", total: 24, active: 18, pending: 4, issues: 2 },
  { type: "Drivers", total: 847, active: 632, pending: 89, issues: 12 },
  { type: "Vehicles", total: 312, active: 287, pending: 15, issues: 8 },
];

const alertColors = {
  critical: "border-l-destructive bg-destructive/10",
  warning: "border-l-warning bg-warning/10",
  info: "border-l-cyan bg-cyan/10",
};

const alertIcons = {
  critical: AlertTriangle,
  warning: Clock,
  info: FileText,
};

export default function AdminDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("overview");

  const handleLogout = () => {
    logout();
    navigate("/auth");
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="h-16 border-b border-border bg-card/50 backdrop-blur-xl flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center glow-gold text-xl">
              {user?.avatar}
            </div>
            <div>
              <h1 className="font-display text-lg font-bold text-gradient-gold">OneHub Command</h1>
              <p className="text-xs text-muted-foreground">Admin Portal • Full Ecosystem Access</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search ecosystem..."
              className="w-80 h-10 pl-10 pr-4 bg-secondary/50 border border-border rounded-lg text-sm"
            />
          </div>
          <button className="relative w-10 h-10 rounded-lg bg-destructive/20 flex items-center justify-center">
            <Bell className="w-5 h-5 text-destructive" />
            <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-destructive text-[10px] flex items-center justify-center">12</span>
          </button>
          <button onClick={handleLogout} className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center hover:bg-destructive/20 transition-colors">
            <LogOut className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 h-[calc(100vh-64px)] border-r border-border p-4">
          <nav className="space-y-1">
            {[
              { id: "overview", label: "Ecosystem Overview", icon: Globe },
              { id: "brands", label: "Brand Management", icon: Building2, badge: "5" },
              { id: "vendors", label: "Vendor Oversight", icon: Truck, badge: "24" },
              { id: "drivers", label: "Driver Network", icon: Users, badge: "847" },
              { id: "compliance", label: "Compliance Center", icon: Shield, badge: "12" },
              { id: "analytics", label: "Cross-Analytics", icon: BarChart3 },
              { id: "settings", label: "System Settings", icon: Settings },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                  activeTab === item.id ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-secondary"
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="flex-1 text-left text-sm">{item.label}</span>
                {item.badge && (
                  <span className="px-2 py-0.5 rounded-full bg-secondary text-muted-foreground text-xs">{item.badge}</span>
                )}
              </button>
            ))}
          </nav>

          {/* System Health */}
          <div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-success/10 to-cyan/10 border border-success/20">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-5 h-5 text-success" />
              <span className="font-medium text-sm">System Health</span>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">API Latency</span>
                <span className="text-success font-mono">28ms</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Uptime</span>
                <span className="text-success font-mono">99.99%</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Active Syncs</span>
                <span className="text-cyan font-mono">3,847</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-y-auto">
          {/* Top KPIs */}
          <div className="grid grid-cols-5 gap-4 mb-6">
            {[
              { label: "Active Deliveries", value: "3,847", icon: Zap, color: "primary" },
              { label: "Total Revenue Today", value: "AED 284K", icon: DollarSign, color: "success" },
              { label: "Online Drivers", value: "632", icon: Users, color: "cyan" },
              { label: "Success Rate", value: "98.7%", icon: CheckCircle, color: "success" },
              { label: "Carbon Saved", value: "1.2 tons", icon: Leaf, color: "success" },
            ].map((kpi, index) => (
              <div key={index} className="glass-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <kpi.icon className={cn("w-5 h-5", `text-${kpi.color}`)} />
                  <span className="text-xs text-muted-foreground">{kpi.label}</span>
                </div>
                <p className="text-xl font-display font-bold">{kpi.value}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-3 gap-6">
            {/* Entity Overview */}
            <div className="col-span-2 glass-card">
              <div className="flex items-center justify-between p-4 border-b border-border">
                <h3 className="font-display text-lg font-semibold">Entity Overview</h3>
                <button className="text-sm text-primary">View All →</button>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-4 gap-4">
                  {entitySummaries.map((entity, index) => (
                    <div key={index} className="p-4 rounded-xl bg-secondary/30 border border-border/50">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-medium">{entity.type}</span>
                        {entity.issues > 0 && (
                          <span className="px-2 py-0.5 rounded-full bg-destructive/20 text-destructive text-xs">
                            {entity.issues} issues
                          </span>
                        )}
                      </div>
                      <p className="text-2xl font-display font-bold mb-2">{entity.total}</p>
                      <div className="space-y-1 text-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Active</span>
                          <span className="text-success">{entity.active}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Pending</span>
                          <span className="text-warning">{entity.pending}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* System Alerts */}
            <div className="glass-card">
              <div className="flex items-center justify-between p-4 border-b border-border">
                <h3 className="font-display text-lg font-semibold">System Alerts</h3>
                <span className="px-2 py-0.5 rounded-full bg-destructive/20 text-destructive text-xs">
                  {systemAlerts.filter((a) => a.type === "critical").length} critical
                </span>
              </div>
              <div className="p-4 space-y-3 max-h-80 overflow-y-auto">
                {systemAlerts.map((alert) => {
                  const Icon = alertIcons[alert.type];
                  return (
                    <div key={alert.id} className={cn("p-3 rounded-lg border-l-4", alertColors[alert.type])}>
                      <div className="flex items-start gap-3">
                        <Icon className={cn(
                          "w-5 h-5 mt-0.5",
                          alert.type === "critical" && "text-destructive",
                          alert.type === "warning" && "text-warning",
                          alert.type === "info" && "text-cyan"
                        )} />
                        <div className="flex-1">
                          <p className="text-sm font-medium">{alert.message}</p>
                          <p className="text-xs text-muted-foreground">{alert.entity}</p>
                          <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                        </div>
                        <button className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                          <Eye className="w-4 h-4 text-muted-foreground" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Compliance Quick View */}
          <div className="mt-6 glass-card">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-primary" />
                <h3 className="font-display text-lg font-semibold">Compliance Overview</h3>
              </div>
              <button className="px-4 py-2 rounded-lg bg-primary/10 text-primary text-sm">
                Generate TDRA Report
              </button>
            </div>
            <div className="p-4 grid grid-cols-4 gap-4">
              {[
                { label: "RTA Compliant Vehicles", value: "287/312", percent: 92, status: "good" },
                { label: "Valid Driver Licenses", value: "835/847", percent: 98, status: "good" },
                { label: "Food Safety Certs", value: "156/168", percent: 93, status: "good" },
                { label: "Pending Inspections", value: "23", percent: 0, status: "warning" },
              ].map((item, index) => (
                <div key={index} className="p-4 rounded-xl bg-secondary/30">
                  <p className="text-sm text-muted-foreground mb-2">{item.label}</p>
                  <p className="text-xl font-display font-bold mb-2">{item.value}</p>
                  {item.percent > 0 && (
                    <div className="h-2 rounded-full bg-secondary overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full",
                          item.percent >= 95 ? "bg-success" : item.percent >= 80 ? "bg-warning" : "bg-destructive"
                        )}
                        style={{ width: `${item.percent}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
