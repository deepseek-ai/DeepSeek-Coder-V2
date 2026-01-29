import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Building2,
  Truck,
  Users,
  Wallet,
  Shield,
  BarChart3,
  MessageSquare,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  Globe,
} from "lucide-react";

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
  badge?: string | number;
  badgeType?: "default" | "warning" | "success";
}

const navItems: NavItem[] = [
  { id: "dashboard", label: "Command Nexus", icon: LayoutDashboard },
  { id: "brands", label: "Brand Hub", icon: Building2, badge: "5", badgeType: "default" },
  { id: "vendors", label: "Vendor Fleet", icon: Truck, badge: "12", badgeType: "success" },
  { id: "drivers", label: "Driver Network", icon: Users, badge: "847", badgeType: "default" },
  { id: "financials", label: "Financial Ledger", icon: Wallet },
  { id: "compliance", label: "Compliance", icon: Shield, badge: "3", badgeType: "warning" },
  { id: "analytics", label: "Analytics AI", icon: BarChart3 },
  { id: "communications", label: "Comms Center", icon: MessageSquare, badge: "24", badgeType: "default" },
];

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeItem, setActiveItem] = useState("dashboard");

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 h-screen bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300 z-50",
        isCollapsed ? "w-20" : "w-64"
      )}
    >
      {/* Logo Section */}
      <div className="h-20 flex items-center justify-between px-4 border-b border-sidebar-border">
        <div className={cn("flex items-center gap-3", isCollapsed && "justify-center")}>
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center glow-gold">
            <Globe className="w-5 h-5 text-primary" />
          </div>
          {!isCollapsed && (
            <div className="animate-fade-in-up">
              <h1 className="font-display text-sm font-bold text-gradient-gold">OneHubDeliOps</h1>
              <p className="text-[10px] text-muted-foreground">UAE Logistics Hub</p>
            </div>
          )}
        </div>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="w-8 h-8 rounded-lg bg-secondary/50 flex items-center justify-center hover:bg-secondary transition-colors"
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          ) : (
            <ChevronLeft className="w-4 h-4 text-muted-foreground" />
          )}
        </button>
      </div>

      {/* Live Status */}
      <div className={cn("px-4 py-4 border-b border-sidebar-border", isCollapsed && "px-2")}>
        <div className={cn("glass-card p-3", isCollapsed && "p-2")}>
          <div className="flex items-center gap-2">
            <div className="status-online" />
            {!isCollapsed && (
              <div className="animate-fade-in-up">
                <p className="text-xs text-success font-medium">System Online</p>
                <p className="text-[10px] text-muted-foreground">All services operational</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
        {navItems.map((item, index) => (
          <button
            key={item.id}
            onClick={() => setActiveItem(item.id)}
            className={cn(
              "w-full nav-item group",
              activeItem === item.id && "active",
              "animate-slide-in-left"
            )}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <item.icon
              className={cn(
                "w-5 h-5 transition-colors",
                activeItem === item.id ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
              )}
            />
            {!isCollapsed && (
              <>
                <span className="flex-1 text-left text-sm">{item.label}</span>
                {item.badge && (
                  <span
                    className={cn(
                      "kpi-badge",
                      item.badgeType === "success" && "positive",
                      item.badgeType === "warning" && "bg-warning/20 text-warning",
                      item.badgeType === "default" && "bg-secondary text-muted-foreground"
                    )}
                  >
                    {item.badge}
                  </span>
                )}
              </>
            )}
          </button>
        ))}
      </nav>

      {/* Quick Actions */}
      <div className="p-4 border-t border-sidebar-border space-y-2">
        <button
          className={cn(
            "w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-all",
            isCollapsed && "justify-center px-2"
          )}
        >
          <Zap className="w-5 h-5" />
          {!isCollapsed && <span className="text-sm font-medium">Quick Deploy</span>}
        </button>
        <button
          className={cn(
            "w-full nav-item",
            isCollapsed && "justify-center px-2"
          )}
        >
          <Settings className="w-5 h-5 text-muted-foreground" />
          {!isCollapsed && <span className="text-sm">Settings</span>}
        </button>
      </div>
    </aside>
  );
}
