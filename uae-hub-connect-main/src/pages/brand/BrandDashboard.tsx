import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import {
  Building2,
  Package,
  TrendingUp,
  Clock,
  DollarSign,
  Users,
  BarChart3,
  Settings,
  Bell,
  Search,
  LogOut,
  ChevronDown,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  RefreshCw,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Order {
  id: string;
  customer: string;
  destination: string;
  status: "pending" | "assigned" | "in_transit" | "delivered";
  vendor: string;
  driver: string;
  eta: string;
  value: number;
}

const mockOrders: Order[] = [
  { id: "ORD-7841", customer: "Ahmed S.", destination: "Dubai Marina", status: "in_transit", vendor: "QuickFleet", driver: "Mohammed K.", eta: "12 min", value: 145 },
  { id: "ORD-7842", customer: "Fatima H.", destination: "JBR Walk", status: "assigned", vendor: "SwiftDel", driver: "Raj P.", eta: "18 min", value: 89 },
  { id: "ORD-7843", customer: "Omar L.", destination: "Downtown Dubai", status: "pending", vendor: "-", driver: "-", eta: "-", value: 234 },
  { id: "ORD-7844", customer: "Sara A.", destination: "Business Bay", status: "delivered", vendor: "QuickFleet", driver: "Ahmed R.", eta: "Done", value: 178 },
  { id: "ORD-7845", customer: "Khalid M.", destination: "DIFC", status: "in_transit", vendor: "SwiftDel", driver: "Hassan A.", eta: "8 min", value: 312 },
];

const statusColors = {
  pending: "bg-warning/20 text-warning",
  assigned: "bg-cyan/20 text-cyan",
  in_transit: "bg-primary/20 text-primary",
  delivered: "bg-success/20 text-success",
};

export default function BrandDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("orders");

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
            <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center text-xl">
              {user?.avatar}
            </div>
            <div>
              <h1 className="font-display text-lg font-bold text-gradient-gold">{user?.company}</h1>
              <p className="text-xs text-muted-foreground">Brand Portal</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search orders..."
              className="w-64 h-10 pl-10 pr-4 bg-secondary/50 border border-border rounded-lg text-sm"
            />
          </div>
          <button className="relative w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
            <Bell className="w-5 h-5 text-muted-foreground" />
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-destructive text-[10px] flex items-center justify-center">5</span>
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
              { id: "orders", label: "Live Orders", icon: Package, badge: "127" },
              { id: "vendors", label: "Vendor Assignment", icon: Users },
              { id: "analytics", label: "Analytics", icon: BarChart3 },
              { id: "financials", label: "Financials", icon: DollarSign },
              { id: "settings", label: "Settings", icon: Settings },
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
                  <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs">{item.badge}</span>
                )}
              </button>
            ))}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {/* KPIs */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            {[
              { label: "Today's Orders", value: "1,247", change: 12.5, icon: Package, color: "primary" },
              { label: "Active Deliveries", value: "342", change: 8.2, icon: TrendingUp, color: "cyan" },
              { label: "Avg. Delivery Time", value: "24 min", change: -5.3, icon: Clock, color: "success" },
              { label: "Revenue Today", value: "AED 89K", change: 18.7, icon: DollarSign, color: "gold" },
            ].map((kpi, index) => (
              <div key={index} className="glass-card p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", `bg-${kpi.color}/20`)}>
                    <kpi.icon className={cn("w-5 h-5", `text-${kpi.color}`)} />
                  </div>
                  <div className={cn("flex items-center gap-1 text-sm", kpi.change > 0 ? "text-success" : "text-destructive")}>
                    {kpi.change > 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                    {Math.abs(kpi.change)}%
                  </div>
                </div>
                <p className="text-2xl font-display font-bold">{kpi.value}</p>
                <p className="text-sm text-muted-foreground">{kpi.label}</p>
              </div>
            ))}
          </div>

          {/* Orders Table */}
          <div className="glass-card">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="font-display text-lg font-semibold">Live Orders</h3>
              <div className="flex items-center gap-2">
                <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary text-sm">
                  <Filter className="w-4 h-4" />
                  Filter
                </button>
                <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/10 text-primary text-sm">
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="p-4 text-sm font-medium text-muted-foreground">Order ID</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Customer</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Destination</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Status</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Vendor</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Driver</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">ETA</th>
                    <th className="p-4 text-sm font-medium text-muted-foreground">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {mockOrders.map((order) => (
                    <tr key={order.id} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                      <td className="p-4 font-mono text-sm text-primary">{order.id}</td>
                      <td className="p-4 text-sm">{order.customer}</td>
                      <td className="p-4 text-sm text-muted-foreground">{order.destination}</td>
                      <td className="p-4">
                        <span className={cn("px-2 py-1 rounded-full text-xs capitalize", statusColors[order.status])}>
                          {order.status.replace("_", " ")}
                        </span>
                      </td>
                      <td className="p-4 text-sm">{order.vendor}</td>
                      <td className="p-4 text-sm">{order.driver}</td>
                      <td className="p-4 text-sm font-medium">{order.eta}</td>
                      <td className="p-4 text-sm font-medium">AED {order.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
