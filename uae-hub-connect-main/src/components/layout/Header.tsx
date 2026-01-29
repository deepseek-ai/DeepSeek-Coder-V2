import { Bell, Search, User, ChevronDown, Wifi, Clock } from "lucide-react";
import { useState, useEffect } from "react";

export function Header() {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-AE", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-AE", {
      weekday: "short",
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  };

  return (
    <header className="h-20 border-b border-border bg-card/50 backdrop-blur-xl flex items-center justify-between px-8">
      {/* Search */}
      <div className="flex-1 max-w-xl">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search entities, orders, drivers..."
            className="w-full h-12 pl-12 pr-4 bg-secondary/50 border border-border rounded-xl text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
          />
          <kbd className="absolute right-4 top-1/2 -translate-y-1/2 px-2 py-1 bg-muted rounded text-[10px] text-muted-foreground font-mono">
            ⌘K
          </kbd>
        </div>
      </div>

      {/* Center Stats */}
      <div className="hidden lg:flex items-center gap-8 px-8">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          <span className="text-sm text-muted-foreground">847 Drivers Active</span>
        </div>
        <div className="h-6 w-px bg-border" />
        <div className="flex items-center gap-2">
          <Wifi className="w-4 h-4 text-cyan" />
          <span className="text-sm text-muted-foreground">Sync: 28ms</span>
        </div>
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-4">
        {/* Time Display */}
        <div className="hidden md:flex flex-col items-end">
          <span className="font-display text-lg text-primary">{formatTime(currentTime)}</span>
          <span className="text-xs text-muted-foreground">{formatDate(currentTime)} • Dubai, UAE</span>
        </div>

        <div className="h-10 w-px bg-border" />

        {/* Notifications */}
        <button className="relative w-10 h-10 rounded-xl bg-secondary/50 flex items-center justify-center hover:bg-secondary transition-colors">
          <Bell className="w-5 h-5 text-muted-foreground" />
          <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-destructive text-[10px] font-bold flex items-center justify-center text-destructive-foreground">
            7
          </span>
        </button>

        {/* User Profile */}
        <button className="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-secondary/50 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
            <User className="w-5 h-5 text-primary" />
          </div>
          <div className="hidden md:block text-left">
            <p className="text-sm font-medium">Admin Portal</p>
            <p className="text-xs text-muted-foreground">Super Admin</p>
          </div>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </button>
      </div>
    </header>
  );
}
