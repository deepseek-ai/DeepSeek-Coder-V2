import { cn } from "@/lib/utils";
import { Trophy, Star, TrendingUp, Award } from "lucide-react";

interface Driver {
  id: string;
  name: string;
  avatar: string;
  rating: number;
  deliveriesToday: number;
  efficiency: number;
  rank: number;
  badge?: "gold" | "silver" | "bronze";
}

const topDrivers: Driver[] = [
  { id: "1", name: "Ahmed Al-Rashid", avatar: "👨🏽", rating: 4.98, deliveriesToday: 47, efficiency: 99.2, rank: 1, badge: "gold" },
  { id: "2", name: "Fatima Hassan", avatar: "👩🏽", rating: 4.96, deliveriesToday: 45, efficiency: 98.8, rank: 2, badge: "silver" },
  { id: "3", name: "Mohammed Khan", avatar: "👨🏾", rating: 4.95, deliveriesToday: 43, efficiency: 98.5, rank: 3, badge: "bronze" },
  { id: "4", name: "Sarah Ibrahim", avatar: "👩🏻", rating: 4.93, deliveriesToday: 41, efficiency: 97.9, rank: 4 },
  { id: "5", name: "Raj Patel", avatar: "👨🏽", rating: 4.91, deliveriesToday: 39, efficiency: 97.5, rank: 5 },
];

const badgeColors = {
  gold: "bg-primary/20 text-primary border-primary/50",
  silver: "bg-slate-400/20 text-slate-300 border-slate-400/50",
  bronze: "bg-amber-700/20 text-amber-500 border-amber-700/50",
};

export function DriverLeaderboard() {
  return (
    <div className="glass-card p-6 animate-fade-in-up" style={{ animationDelay: "500ms" }}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
            <Trophy className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-gradient-gold">Top Performers</h3>
            <p className="text-sm text-muted-foreground">Today's driver leaderboard</p>
          </div>
        </div>
        <button className="text-sm text-primary hover:underline">View All →</button>
      </div>

      <div className="space-y-3">
        {topDrivers.map((driver, index) => (
          <div
            key={driver.id}
            className={cn(
              "flex items-center gap-4 p-4 rounded-xl bg-secondary/30 border border-border/30 hover:border-primary/30 transition-all cursor-pointer",
              index === 0 && "ring-1 ring-primary/30 bg-primary/5",
              "animate-slide-in-right"
            )}
            style={{ animationDelay: `${500 + index * 100}ms` }}
          >
            {/* Rank */}
            <div
              className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center font-display font-bold text-sm",
                driver.badge ? badgeColors[driver.badge] + " border" : "bg-secondary text-muted-foreground"
              )}
            >
              {driver.rank}
            </div>

            {/* Avatar */}
            <div className="w-10 h-10 rounded-full bg-card flex items-center justify-center text-2xl">
              {driver.avatar}
            </div>

            {/* Driver Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="font-medium text-foreground truncate">{driver.name}</h4>
                {driver.badge && (
                  <Award
                    className={cn(
                      "w-4 h-4",
                      driver.badge === "gold" && "text-primary",
                      driver.badge === "silver" && "text-slate-300",
                      driver.badge === "bronze" && "text-amber-500"
                    )}
                  />
                )}
              </div>
              <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Star className="w-3 h-3 text-primary fill-primary" />
                  {driver.rating}
                </span>
                <span>{driver.deliveriesToday} deliveries</span>
              </div>
            </div>

            {/* Efficiency */}
            <div className="text-right">
              <div className="flex items-center gap-1 justify-end">
                <span className="font-display text-lg font-bold text-success">{driver.efficiency}%</span>
                <TrendingUp className="w-4 h-4 text-success" />
              </div>
              <p className="text-xs text-muted-foreground">Efficiency</p>
            </div>
          </div>
        ))}
      </div>

      {/* Incentive Banner */}
      <div className="mt-4 p-4 rounded-xl bg-gradient-to-r from-primary/10 to-cyan/10 border border-primary/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-foreground">🎯 Daily Challenge Active</p>
            <p className="text-xs text-muted-foreground">Top 10 drivers earn AED 500 bonus</p>
          </div>
          <span className="font-display text-primary font-bold">4h 23m left</span>
        </div>
      </div>
    </div>
  );
}
