import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import {
  Truck,
  Users,
  DollarSign,
  TrendingUp,
  Clock,
  MapPin,
  Bell,
  LogOut,
  Search,
  Gavel,
  Wallet,
  Settings,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  Car,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Driver {
  id: string;
  name: string;
  avatar: string;
  status: "online" | "busy" | "offline";
  currentLocation: string;
  deliveriesToday: number;
  rating: number;
  vehicle: string;
}

interface Bid {
  id: string;
  brand: string;
  ordersCount: number;
  zone: string;
  basePrice: number;
  currentBid: number;
  timeLeft: string;
  status: "active" | "won" | "lost";
}

const mockDrivers: Driver[] = [
  { id: "D001", name: "Mohammed Khan", avatar: "👨🏾", status: "busy", currentLocation: "Dubai Marina", deliveriesToday: 12, rating: 4.95, vehicle: "Bike" },
  { id: "D002", name: "Raj Patel", avatar: "👨🏽", status: "online", currentLocation: "JBR", deliveriesToday: 8, rating: 4.88, vehicle: "Car" },
  { id: "D003", name: "Hassan Ali", avatar: "👨🏻", status: "online", currentLocation: "Downtown", deliveriesToday: 15, rating: 4.92, vehicle: "Bike" },
  { id: "D004", name: "Ahmed Rashid", avatar: "👨🏽", status: "offline", currentLocation: "-", deliveriesToday: 0, rating: 4.78, vehicle: "Van" },
];

const mockBids: Bid[] = [
  { id: "BID-001", brand: "Noon", ordersCount: 50, zone: "Dubai Marina", basePrice: 2500, currentBid: 2200, timeLeft: "2:34", status: "active" },
  { id: "BID-002", brand: "Careem", ordersCount: 30, zone: "Downtown", basePrice: 1800, currentBid: 1650, timeLeft: "5:12", status: "active" },
  { id: "BID-003", brand: "Amazon", ordersCount: 75, zone: "JBR", basePrice: 3500, currentBid: 3200, timeLeft: "-", status: "won" },
];

const statusColors = {
  online: "bg-success",
  busy: "bg-warning",
  offline: "bg-muted-foreground",
};

export default function VendorDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("fleet");

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
            <div className="w-10 h-10 rounded-xl bg-cyan/20 flex items-center justify-center text-xl">
              {user?.avatar}
            </div>
            <div>
              <h1 className="font-display text-lg font-bold text-gradient-gold">{user?.company}</h1>
              <p className="text-xs text-muted-foreground">Vendor Portal</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button className="relative w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
            <Bell className="w-5 h-5 text-muted-foreground" />
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-primary text-[10px] flex items-center justify-center">3</span>
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
              { id: "fleet", label: "Fleet Overview", icon: Truck, badge: "12" },
              { id: "drivers", label: "Driver Management", icon: Users },
              { id: "bidding", label: "Active Bids", icon: Gavel, badge: "2" },
              { id: "earnings", label: "Earnings", icon: Wallet },
              { id: "settings", label: "Settings", icon: Settings },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                  activeTab === item.id ? "bg-cyan/10 text-cyan" : "text-muted-foreground hover:bg-secondary"
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="flex-1 text-left text-sm">{item.label}</span>
                {item.badge && (
                  <span className="px-2 py-0.5 rounded-full bg-cyan/20 text-cyan text-xs">{item.badge}</span>
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
              { label: "Active Drivers", value: "12", subtext: "3 online, 9 busy", icon: Users, color: "cyan" },
              { label: "Today's Earnings", value: "AED 4.2K", subtext: "+18% vs yesterday", icon: DollarSign, color: "success" },
              { label: "Deliveries Today", value: "89", subtext: "42 remaining", icon: Truck, color: "primary" },
              { label: "Fleet Utilization", value: "87%", subtext: "Optimal range", icon: TrendingUp, color: "gold" },
            ].map((kpi, index) => (
              <div key={index} className="glass-card p-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", `bg-${kpi.color}/20`)}>
                    <kpi.icon className={cn("w-5 h-5", `text-${kpi.color}`)} />
                  </div>
                </div>
                <p className="text-2xl font-display font-bold">{kpi.value}</p>
                <p className="text-sm text-muted-foreground">{kpi.label}</p>
                <p className="text-xs text-muted-foreground mt-1">{kpi.subtext}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Driver Fleet */}
            <div className="glass-card">
              <div className="flex items-center justify-between p-4 border-b border-border">
                <h3 className="font-display text-lg font-semibold">Driver Fleet</h3>
                <button className="text-sm text-cyan">Manage All →</button>
              </div>
              <div className="p-4 space-y-3">
                {mockDrivers.map((driver) => (
                  <div key={driver.id} className="flex items-center gap-4 p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full bg-card flex items-center justify-center text-2xl">
                        {driver.avatar}
                      </div>
                      <div className={cn("absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full border-2 border-card", statusColors[driver.status])} />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">{driver.name}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <MapPin className="w-3 h-3" />
                        {driver.currentLocation}
                        <span>•</span>
                        <Car className="w-3 h-3" />
                        {driver.vehicle}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-display font-bold text-success">{driver.deliveriesToday}</p>
                      <p className="text-xs text-muted-foreground">deliveries</p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-muted-foreground" />
                  </div>
                ))}
              </div>
            </div>

            {/* Active Bids */}
            <div className="glass-card">
              <div className="flex items-center justify-between p-4 border-b border-border">
                <h3 className="font-display text-lg font-semibold">Active Bidding</h3>
                <button className="text-sm text-cyan">View History →</button>
              </div>
              <div className="p-4 space-y-3">
                {mockBids.map((bid) => (
                  <div key={bid.id} className={cn(
                    "p-4 rounded-lg border transition-colors",
                    bid.status === "active" ? "bg-secondary/30 border-cyan/30" : bid.status === "won" ? "bg-success/10 border-success/30" : "bg-card border-border"
                  )}>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="font-display font-bold">{bid.brand}</span>
                        <span className="px-2 py-0.5 rounded-full bg-secondary text-xs">{bid.ordersCount} orders</span>
                      </div>
                      {bid.status === "active" ? (
                        <span className="text-sm font-mono text-cyan">{bid.timeLeft} left</span>
                      ) : bid.status === "won" ? (
                        <CheckCircle className="w-5 h-5 text-success" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-muted-foreground" />
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">{bid.zone} Zone</p>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-muted-foreground">Current Bid</p>
                        <p className="font-display font-bold text-lg">AED {bid.currentBid}</p>
                      </div>
                      {bid.status === "active" && (
                        <button className="px-4 py-2 rounded-lg bg-cyan text-background font-medium text-sm hover:bg-cyan/90 transition-colors">
                          Place Bid
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
