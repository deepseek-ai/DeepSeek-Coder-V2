import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import {
  Package,
  Navigation,
  Wallet,
  FileText,
  Star,
  Clock,
  MapPin,
  Phone,
  CheckCircle,
  ChevronRight,
  Bell,
  LogOut,
  Trophy,
  Zap,
  TrendingUp,
  User,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Delivery {
  id: string;
  customer: string;
  phone: string;
  address: string;
  items: number;
  value: number;
  eta: string;
  distance: string;
  priority: "normal" | "express";
  status: "current" | "upcoming" | "completed";
}

const mockDeliveries: Delivery[] = [
  { id: "DEL-001", customer: "Ahmed Al-Rashid", phone: "+971 50 123 4567", address: "Tower 3, Apt 1205, Dubai Marina", items: 3, value: 245, eta: "8 min", distance: "2.3 km", priority: "express", status: "current" },
  { id: "DEL-002", customer: "Fatima Hassan", phone: "+971 55 987 6543", address: "Villa 24, Palm Jumeirah", items: 1, value: 89, eta: "25 min", distance: "5.1 km", priority: "normal", status: "upcoming" },
  { id: "DEL-003", customer: "Omar Khalid", phone: "+971 52 456 7890", address: "Office 405, DIFC Gate Village", items: 5, value: 412, eta: "45 min", distance: "8.2 km", priority: "normal", status: "upcoming" },
];

export default function DriverDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("deliveries");

  const handleLogout = () => {
    logout();
    navigate("/auth");
  };

  const currentDelivery = mockDeliveries.find((d) => d.status === "current");
  const upcomingDeliveries = mockDeliveries.filter((d) => d.status === "upcoming");

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Header */}
      <header className="h-16 border-b border-border bg-card/50 backdrop-blur-xl flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-success/20 flex items-center justify-center text-xl">
            {user?.avatar}
          </div>
          <div>
            <p className="font-medium text-sm">{user?.name}</p>
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <div className="w-2 h-2 rounded-full bg-success" />
              Online • Dubai Marina
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="relative w-10 h-10 rounded-full bg-secondary/50 flex items-center justify-center">
            <Bell className="w-5 h-5 text-muted-foreground" />
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-destructive text-[9px] flex items-center justify-center">2</span>
          </button>
          <button onClick={handleLogout} className="w-10 h-10 rounded-full bg-secondary/50 flex items-center justify-center">
            <LogOut className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>
      </header>

      {/* Today's Stats */}
      <div className="p-4">
        <div className="glass-card p-4 mb-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display font-semibold">Today's Performance</h3>
            <div className="flex items-center gap-1 text-success text-sm">
              <TrendingUp className="w-4 h-4" />
              +12%
            </div>
          </div>
          <div className="grid grid-cols-4 gap-3">
            <div className="text-center">
              <p className="text-2xl font-display font-bold text-primary">12</p>
              <p className="text-xs text-muted-foreground">Deliveries</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-display font-bold text-success">AED 280</p>
              <p className="text-xs text-muted-foreground">Earnings</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-display font-bold text-cyan">4.98</p>
              <p className="text-xs text-muted-foreground">Rating</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-display font-bold text-foreground">24m</p>
              <p className="text-xs text-muted-foreground">Avg. Time</p>
            </div>
          </div>
        </div>

        {/* Current Delivery */}
        {currentDelivery && (
          <div className="mb-4">
            <h3 className="font-display font-semibold mb-3 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
              Current Delivery
            </h3>
            <div className="glass-card p-4 border-l-4 border-l-success">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-medium">
                    {currentDelivery.priority.toUpperCase()}
                  </span>
                  <span className="font-mono text-sm text-muted-foreground">{currentDelivery.id}</span>
                </div>
                <span className="font-display font-bold text-success">{currentDelivery.eta}</span>
              </div>
              
              <div className="flex items-start gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
                  <User className="w-5 h-5 text-muted-foreground" />
                </div>
                <div className="flex-1">
                  <p className="font-medium">{currentDelivery.customer}</p>
                  <p className="text-sm text-muted-foreground">{currentDelivery.address}</p>
                  <p className="text-xs text-muted-foreground mt-1">{currentDelivery.items} items • AED {currentDelivery.value}</p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <button className="flex items-center justify-center gap-2 py-3 rounded-lg bg-cyan text-background font-medium">
                  <Navigation className="w-4 h-4" />
                  Navigate
                </button>
                <button className="flex items-center justify-center gap-2 py-3 rounded-lg bg-secondary">
                  <Phone className="w-4 h-4" />
                  Call
                </button>
                <button className="flex items-center justify-center gap-2 py-3 rounded-lg bg-success text-success-foreground font-medium">
                  <CheckCircle className="w-4 h-4" />
                  Complete
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Upcoming Deliveries */}
        <div className="mb-4">
          <h3 className="font-display font-semibold mb-3">Upcoming ({upcomingDeliveries.length})</h3>
          <div className="space-y-3">
            {upcomingDeliveries.map((delivery) => (
              <div key={delivery.id} className="glass-card p-4 flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center">
                  <Package className="w-6 h-6 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{delivery.customer}</p>
                  <p className="text-sm text-muted-foreground truncate">{delivery.address}</p>
                  <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {delivery.eta}
                    <span>•</span>
                    <MapPin className="w-3 h-3" />
                    {delivery.distance}
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-display font-bold">AED {delivery.value}</p>
                  <p className="text-xs text-muted-foreground">{delivery.items} items</p>
                </div>
                <ChevronRight className="w-5 h-5 text-muted-foreground" />
              </div>
            ))}
          </div>
        </div>

        {/* Daily Challenge */}
        <div className="glass-card p-4 bg-gradient-to-r from-primary/10 to-cyan/10 border border-primary/20">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
              <Trophy className="w-6 h-6 text-primary" />
            </div>
            <div className="flex-1">
              <p className="font-medium">Daily Challenge</p>
              <p className="text-sm text-muted-foreground">Complete 15 deliveries for AED 50 bonus</p>
            </div>
            <div className="text-right">
              <p className="font-display font-bold text-primary">12/15</p>
              <p className="text-xs text-muted-foreground">3 more</p>
            </div>
          </div>
          <div className="mt-3 h-2 rounded-full bg-secondary overflow-hidden">
            <div className="h-full rounded-full bg-primary" style={{ width: "80%" }} />
          </div>
        </div>
      </div>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 h-16 border-t border-border bg-card/95 backdrop-blur-xl flex items-center justify-around px-4">
        {[
          { id: "deliveries", label: "Deliveries", icon: Package },
          { id: "earnings", label: "Earnings", icon: Wallet },
          { id: "documents", label: "Documents", icon: FileText },
          { id: "profile", label: "Profile", icon: User },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={cn(
              "flex flex-col items-center gap-1 py-2 px-4 rounded-lg transition-colors",
              activeTab === item.id ? "text-primary" : "text-muted-foreground"
            )}
          >
            <item.icon className="w-5 h-5" />
            <span className="text-xs">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
