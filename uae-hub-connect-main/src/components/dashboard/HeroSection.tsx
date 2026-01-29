import { useEffect, useState } from "react";
import { Globe, Zap, Shield, BarChart3 } from "lucide-react";

export function HeroSection() {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([]);

  useEffect(() => {
    const newParticles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 5,
    }));
    setParticles(newParticles);
  }, []);

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-secondary/50 to-card border border-border/50 mb-8">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-pattern opacity-20" />
      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-cyan/5" />

      {/* Floating Particles */}
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute w-1 h-1 rounded-full bg-primary/40 animate-float"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            animationDelay: `${particle.delay}s`,
          }}
        />
      ))}

      {/* Orbital Rings */}
      <div className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2">
        <div className="w-64 h-64 rounded-full border border-primary/10 animate-orbit-slow" />
        <div className="absolute inset-4 rounded-full border border-cyan/10 animate-orbit-reverse" />
        <div className="absolute inset-8 rounded-full border border-primary/5 animate-orbit" />
      </div>

      <div className="relative z-10 p-8 md:p-12">
        <div className="max-w-2xl">
          {/* Status Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-success/10 border border-success/30 mb-6 animate-fade-in-up">
            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <span className="text-sm text-success font-medium">All Systems Operational</span>
            <span className="text-xs text-muted-foreground">• UAE Region</span>
          </div>

          {/* Main Title */}
          <h1 className="font-display text-4xl md:text-5xl font-bold mb-4 animate-fade-in-up" style={{ animationDelay: "100ms" }}>
            <span className="text-gradient-gold">OneHubDeliOps</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-6 animate-fade-in-up" style={{ animationDelay: "200ms" }}>
            The Unified Delivery Ecosystem for UAE's Logistics Revolution
          </p>

          {/* Quick Stats Row */}
          <div className="flex flex-wrap gap-6 mb-8 animate-fade-in-up" style={{ animationDelay: "300ms" }}>
            <div className="flex items-center gap-2">
              <Globe className="w-5 h-5 text-cyan" />
              <span className="text-sm"><span className="font-bold text-foreground">5</span> <span className="text-muted-foreground">Connected Brands</span></span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              <span className="text-sm"><span className="font-bold text-foreground">28ms</span> <span className="text-muted-foreground">Sync Speed</span></span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-success" />
              <span className="text-sm"><span className="font-bold text-foreground">99.99%</span> <span className="text-muted-foreground">Uptime</span></span>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-info" />
              <span className="text-sm"><span className="font-bold text-foreground">2000x</span> <span className="text-muted-foreground">Efficiency Gain</span></span>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-4 animate-fade-in-up" style={{ animationDelay: "400ms" }}>
            <button className="px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity glow-gold">
              Launch Command Center
            </button>
            <button className="px-6 py-3 rounded-xl border border-border text-foreground font-semibold hover:bg-secondary/50 transition-colors">
              View Documentation
            </button>
          </div>
        </div>

        {/* Right Side Decorative Element */}
        <div className="hidden lg:block absolute right-12 top-1/2 -translate-y-1/2">
          <div className="relative w-48 h-48">
            {/* Central Orb */}
            <div className="absolute inset-8 rounded-full bg-gradient-to-br from-primary/30 to-cyan/20 animate-pulse-glow blur-xl" />
            <div className="absolute inset-12 rounded-full bg-gradient-to-br from-primary/50 to-cyan/30 flex items-center justify-center">
              <Globe className="w-12 h-12 text-primary" />
            </div>

            {/* Orbiting Icons */}
            <div className="absolute inset-0 animate-orbit">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <Zap className="w-4 h-4 text-primary" />
              </div>
            </div>
            <div className="absolute inset-0 animate-orbit-reverse">
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                <Shield className="w-4 h-4 text-success" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
